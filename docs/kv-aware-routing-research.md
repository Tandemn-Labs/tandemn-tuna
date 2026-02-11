# KV-Aware Routing Research

Deep research into KV-aware routing for LLM inference, with feasibility analysis
for tuna's hybrid spot/serverless architecture.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Dynamo (NVIDIA)](#2-dynamo-nvidia)
3. [AIBrix (vLLM Project)](#3-aibrix-vllm-project)
4. [Other Novel Routing Strategies](#4-other-novel-routing-strategies)
5. [Feasibility for tuna](#5-feasibility-for-tuna)
6. [Concrete Architecture Vision](#6-concrete-architecture-vision)
7. [Open Questions](#7-open-questions)

---

## 1. The Problem

In standard LLM serving, if a user sends a multi-turn conversation, each request
must recompute the KV cache for the entire conversation prefix. If the router
randomly assigns the request to a different GPU than the one that served the
previous turn, the KV cache from the previous turn is wasted.

This matters even more in a hybrid spot/serverless setup because:
- **Spot preemption** destroys GPU-resident KV cache without warning
- **Serverless cold starts** mean zero cached state
- **Cross-provider routing** (Modal vs RunPod vs Cloud Run vs spot) means no
  shared memory between backends
- Bursty traffic can saturate spot while serverless sits idle with no overflow

The core question: can we build a router that is aware of KV cache state across
heterogeneous, ephemeral backends?

---

## 2. Dynamo (NVIDIA)

GitHub: https://github.com/ai-dynamo/dynamo

### Architecture

Dynamo is a Rust + Python framework for datacenter-scale LLM serving with five
core components:

| Component | Role |
|---|---|
| Frontend | OpenAI-compatible HTTP, tokenization, detokenization |
| Smart Router | KV-cache-aware request routing |
| Disaggregated Serving | Separate prefill and decode worker pools |
| KV Cache Block Manager | Multi-tier memory (GPU -> CPU -> SSD -> remote) |
| Planner | SLO-based autoscaling control plane |

Communication: NATS for events, etcd for service discovery, NIXL for GPU-to-GPU
KV cache transfers.

### How the Router Knows KV Cache State

**Workers emit fine-grained events:**
- `KV Stored`: worker_id, block hashes, token hashes, parent block hash
- `KV Removed`: list of evicted block hashes
- `KV Cleared`: full purge for a worker

**Router builds a global radix tree** (`/lib/kv-router/src/radix_tree.rs`):

```
RadixTree
  root: RadixBlock
    children: HashMap<LocalBlockHash, RadixBlock>
    workers: HashSet<WorkerWithDpRank>     // which workers cache this block
    recent_uses: VecDeque<Instant>          // frequency tracking
  lookup: HashMap<Worker, HashMap<BlockHash, RadixBlock>>  // O(1) worker->block
```

Each node in the tree represents a block of tokens. Edges are labeled with token
block hashes. The `workers` set at each node tells the router exactly which GPUs
hold that prefix.

### The Cost-Based Routing Algorithm

For each candidate worker, the router computes:

```
cost = overlap_weight * new_prefill_blocks + active_decode_blocks
```

Where:
- `new_prefill_blocks` = tokens NOT already cached on this worker (need fresh compute)
- `active_decode_blocks` = blocks in use by ongoing decodes (load proxy)
- `overlap_weight` (default 1.0) = cache reuse vs load balance trade-off

**Lowest cost wins.** Optional softmax temperature for probabilistic selection to
prevent herding (all requests going to one warm-cache worker).

### Handling Instance Failures

- Workers register in etcd with 10-second TTL leases
- If worker crashes, lease expires, router removes all its blocks from radix tree
- **Request migration**: accumulated tokens preserved, request re-routed to new
  worker which continues from the exact failure point
- Configurable via `--migration-limit`

### Disaggregated Prefill/Decode

Separate pools: prefill workers (compute-bound) and decode workers (memory-bound).
After prefill, KV cache is transferred GPU-to-GPU via NIXL (RDMA). Without RDMA,
40x performance degradation.

**Results**: 3x TTFT improvement, 2x latency reduction vs random routing.
30% throughput/GPU improvement from disaggregation.

### Key Knobs

| Parameter | Default | Effect |
|---|---|---|
| `--kv-overlap-score-weight` | 1.0 | Cache reuse vs load balance |
| `--router-temperature` | 0.0 | Deterministic vs probabilistic |
| `--router-ttl-secs` | 120 | Cache entry validity duration |
| `--migration-limit` | 0 | Max migrations on failure |

---

## 3. AIBrix (vLLM Project)

GitHub: https://github.com/vllm-project/aibrix

### Architecture

Go + Python, Kubernetes-native. Two planes:

- **Control Plane**: Model adapter controller, autoscaler, GPU optimizer
- **Data Plane**: Request router (Envoy ext_proc), distributed KV cache runtime

Request flow:
```
Client -> Envoy -> ext_proc -> AIBrix Gateway -> Router -> Pod
```

### 19 Routing Algorithms

AIBrix ships with 19 algorithms, including:

**Load-based**: random, least-request, least-load, throughput, least-latency
**Cache-aware**: prefix-cache, prefix-cache-preble, least-kv-cache, least-gpu-cache
**Advanced**: prefill-decode disaggregation, virtual token counter fairness,
  session affinity, SLO-based, bin packing

### The Prefix Cache Router

Core algorithm:

1. **Tokenize** prompt (character-based at 128-byte blocks, or tiktoken at 16-token blocks)
2. **Load imbalance check**: if `max(running) - min(running) > 8`, skip prefix
   matching, route to least-loaded pod
3. **Compute prefix hash chain**: `h0 = xxhash(seed + block0)`,
   `h1 = xxhash(seed + h0 + block1)`, etc. Each hash depends on ALL previous.
4. **Match against index**: for each hash, look up which pods have that block cached
5. **Select pod**: highest match %, with running requests within `mean +/- stdDev*2`
6. **Fallback**: if all candidates overloaded, pick least-loaded pod

### Two Modes of KV State Tracking

**Mode 1: Local tracking (optimistic)**
Router records prefix-to-pod associations based on its own routing decisions.
Assumes the pod cached the tokens. No verification.

**Mode 2: Real-time KV event sync via ZMQ**
vLLM pods publish actual cache events (BlockStored, BlockRemoved, AllBlocksCleared)
over ZMQ PUB/SUB. Gateway subscribes and updates a `SyncPrefixHashTable`.

Events use MessagePack encoding. Sequence tracking detects missed events with
replay requests via DEALER socket. ~1MB/s network overhead per pod.

### Distributed KV Cache Index

```go
SyncPrefixHashTable:
  contextMap: sync.Map                               // model+LoRA -> ContextData
    prefixMap: map[uint64]map[string]*PodInfo         // hash -> pod -> info
    blockIndex: map[engineHash][]aibrixHash           // reverse mapping
```

Two-level structure: Model context (model + LoRA ID) -> prefix hash -> pod set.
~200ns insertions, ~150ns lookups, ~64 bytes per entry.

### Handling Pod Failures

- Kubernetes informer watches pod lifecycle
- On pod deletion: unsubscribe ZMQ, `RemovePrefix` clears all entries for that pod
- ZMQ client auto-reconnects with exponential backoff (1s -> 30s max)
- Routing algorithms fall back to random when metrics unavailable

### Tiered KV Cache Storage

- **L1**: Local engine DRAM (CPU memory on inference pod)
- **L2**: Distributed via InfiniStore (RDMA transport, Redis metadata)

Enables cross-engine KV reuse for L2.

---

## 4. Other Novel Routing Strategies

### 4.1 Disaggregated Prefill/Decode

| System | Venue | Key Idea | Result |
|---|---|---|---|
| **Splitwise** | ISCA 2024 | Separate prefill/decode pools + dynamic mixed pool | 1.4x throughput, 20% cheaper |
| **DistServe** | OSDI 2024 | Co-optimize parallelism per phase, network-aware placement | 7.4x more requests |
| **Mooncake** | 2024 (Kimi) | KV-centric disaggregation, use idle CPU/DRAM/SSD for cache | 5.25x throughput |
| **P/D-Serve** | 2024 (Huawei) | Dynamic P/D ratio adjustment, optimized RDMA transfer | 6.7x throughput |

### 4.2 Prefix-Aware Routing

| System | Key Idea | Result |
|---|---|---|
| **SGLang RadixAttention** | Radix tree maps token sequences to GPU KV cache tensors | 5x throughput |
| **vLLM APC** | Content-based hashing of token blocks for automatic reuse | Integrated in vLLM |
| **Preble** | Global prompt tree for distributed prefix sharing | 1.5-14.5x latency improvement |
| **MemServe** | Elastic memory pool + prompt tree = stateful LLM serving | TTFT + JCT improvement |

### 4.3 Spot-Instance-Aware Serving

**SpotServe** (ASPLOS 2024) is the definitive paper:
- Dynamic parallelism adaptation as instances appear/disappear
- Migration as bipartite graph matching (Kuhn-Munkres algorithm)
- Token-level progress checkpointing (not checkpoint-level)
- Exploits cloud grace periods (30-120s) for orderly migration
- **Result**: 2.4-9.1x P99 tail latency reduction, 54% cost savings

### 4.4 Cost-Aware Routing

| System | Key Idea | Result |
|---|---|---|
| **RouteLLM** | Train routers on preference data to predict when expensive model needed | 85% cost savings at 95% quality |
| **Hybrid LLM** | Route by query difficulty between small/large models | 40% fewer calls to large model |

### 4.5 Scheduling Innovations

| System | Key Idea | Result |
|---|---|---|
| **Sarathi-Serve** | Chunked prefills + stall-free scheduling | 2.6-5.6x capacity |
| **FastServe** | Token-level preemption + skip-join MLFQ | 31.4x throughput |
| **Andes** | QoE-aware scheduling (user-perceived streaming quality) | 4.7x QoE improvement |
| **Llumnix** | Live migration of requests + KV cache between instances | 36% cost savings |

### 4.6 KV Cache Migration and Compression

| System | Key Idea | Result |
|---|---|---|
| **CacheGen** | Custom tensor encoder exploiting KV distributional properties | 3.5-4.3x compression |
| **Infinite-LLM** | Pooled GPU memory across cluster, distributed attention | 1.35-3.4x throughput |

---

## 5. LMCache — The Missing Piece

GitHub: https://github.com/LMCache/LMCache
Docs: https://docs.lmcache.ai

LMCache is an open-source KV cache management layer that plugs into vLLM as a
connector. It changes the feasibility analysis dramatically because it provides
cross-instance KV cache sharing over standard network protocols (Redis, S3, TCP)
with built-in CacheGen compression.

### What LMCache Does

- Stores KV cache in a tiered hierarchy: GPU -> CPU RAM -> local disk -> remote
  (Redis, S3, Valkey, Mooncake, InfiniStore, NIXL, etc.)
- When Instance A prefills a prompt, KV chunks are stored remotely. When Instance B
  gets the same prefix, it retrieves cached chunks instead of recomputing.
- CacheGen compression (3.5-4.3x) built in, activated with `remote_serde: "cachegen"`
- Prefix-based chunked caching: tokens split into blocks (default 256), hashed with
  cumulative SHA-256 chain, longest contiguous prefix match on retrieval
- CacheBlend: non-prefix reuse (for RAG) — recomputes a configurable subset of tokens
  at non-prefix positions

### Integration With vLLM

LMCache is a vLLM KV connector plugin. Not a fork, not a sidecar:

```bash
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=5 \
LMCACHE_REMOTE_URL="redis://shared-redis:6379" \
LMCACHE_REMOTE_SERDE="cachegen" \
vllm serve Qwen/Qwen3-8B \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

That's it. Environment variables + one CLI flag.

### 14 Storage Backends

| Backend | Transport | Cross-machine? |
|---|---|---|
| CPU RAM | Local | No (per-instance) |
| Local Disk | Filesystem | No (per-instance) |
| GDS (GPU Direct Storage) | cuFile | No (per-instance) |
| Redis / Valkey | TCP | **Yes** |
| S3 | AWS CRT HTTP | **Yes** |
| LMCache Server | TCP sockets | **Yes** |
| Mooncake | TCP/RDMA | **Yes** |
| InfiniStore | RDMA | Yes (same network) |
| NIXL (P2P) | RDMA/TCP | Yes (same network) |
| Weka (WekaFS) | GDS over WekaFS | Yes (shared FS) |
| SageMaker HyperPod | HTTP | Yes (AWS) |
| EIC | RDMA/GDR | Yes (Volcano Engine) |

### Controller + Routing Support

LMCache includes a Controller with REST API:

| Endpoint | What it does |
|---|---|
| `GET /lookup` | Query which instances have cached which token prefixes. Returns `{instance: (backend, matched_prefix_length)}` |
| `POST /compress` | Compress cached KV with CacheGen |
| `POST /move` | Move/copy KV cache between instances |
| `POST /pin` | Pin cache chunks to prevent eviction |
| `POST /clear` | Remove KV cache data |
| `GET /health` | Health check |

The `/lookup` endpoint is the key enabler for KV-aware routing — an external
router can query it to find which backend has the best cache hit before routing.

LMCache also publishes `BlockStored` events via ZMQ that external routers can
consume.

### Performance Numbers

| Scenario | Improvement |
|---|---|
| CPU offloading (single instance) | 7.4x TTFT speedup, 44.5x reported peak |
| Local disk offloading | 42.6x TTFT speedup |
| Redis shared cache (multi-instance) | 75% TTFT reduction |
| P2P sharing via NIXL | 54.7% TTFT reduction, 2.03 GB/s transfer |
| CacheGen compression | 3.5-4.3x cache size reduction |

---

## 6. Feasibility for tuna (Revised With LMCache)

### Key Realization: tuna Controls vLLM on ALL Providers

This was previously underestimated. tuna deploys its own vLLM process on every
provider:

| Provider | Image Control | vLLM Args Control | Env Vars Control |
|---|---|---|---|
| **Modal** | Full (custom image build) | Full (shell string) | Yes (env dict) |
| **Cloud Run** | Yes (official vLLM image) | Full (args list) | Full (env dict) |
| **RunPod** | No (RunPod image) | No (image reads env) | Full (env dict) |
| **SkyPilot spot** | N/A (VMs) | Full (shell cmd in YAML) | Needs small fix |

This means LMCache can be injected into every provider via environment variables:

```
LMCACHE_CHUNK_SIZE=256
LMCACHE_LOCAL_CPU=True
LMCACHE_MAX_LOCAL_CPU_SIZE=5
LMCACHE_REMOTE_URL=redis://shared-redis:6379
LMCACHE_REMOTE_SERDE=cachegen
```

Plus `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
on providers where we control CLI args (Modal, Cloud Run, SkyPilot).

For RunPod: the `runpod/worker-v1-vllm` image reads env vars but may not support
`--kv-transfer-config` directly. We'd need a custom RunPod image with LMCache
pre-installed, OR the RunPod image would need to support LMCache natively. This is
the one provider where injection is harder.

### What's Now Feasible (LMCache Changes Everything)

**1. Cross-provider KV cache sharing (YES — via Redis or S3)**

Previously marked "NOT feasible." Now feasible with LMCache.

All vLLM instances (spot + serverless) connect to a shared Redis or S3 bucket.
When any instance prefills a prompt, the KV chunks are stored remotely with
CacheGen compression (3.5-4.3x). When any other instance — on any provider — gets
the same prefix, it retrieves the cached chunks instead of recomputing.

Architecture:
```
                    ┌──────────────────┐
                    │  Shared Redis /  │
                    │   S3 Bucket      │
                    └───┬──────────┬───┘
                        │          │
            Store KV    │          │  Retrieve KV
            (CacheGen   │          │  (decompress)
             compressed)│          │
                ┌───────┴──┐  ┌────┴───────┐
                │ Spot     │  │ Serverless │
                │ (vLLM +  │  │ (vLLM +    │
                │ LMCache) │  │ LMCache)   │
                └──────────┘  └────────────┘
```

For a 7B model at 4K context: ~1GB KV cache, compressed to ~250MB with CacheGen.
Over a good internet connection, that's 1-2 seconds to retrieve — much faster
than recomputing prefill from scratch (which can take 5-30+ seconds for long
contexts).

The break-even point: LMCache retrieval is faster than recomputation when the
prompt is long enough. For short prompts (<1K tokens), recomputation is fast
anyway. For long contexts (RAG, multi-turn, system prompts), the savings are
massive.

**2. KV-aware routing via LMCache Controller (YES)**

The LMCache Controller's `/lookup` API returns which instances have which prefixes
cached, and how deep the match is. Our router can query this before making a
routing decision:

```
Router receives request
  -> hash the prompt prefix
  -> GET /lookup on LMCache Controller
  -> returns: {spot_1: (cpu, 3840 tokens), modal_a: (redis, 2048 tokens)}
  -> route to spot_1 (deepest match)
```

This gives us Dynamo/AIBrix-style cache-aware routing without building our own
radix tree — LMCache already tracks the state.

**3. Spot preemption with KV cache survival (YES)**

This is the killer feature. When a spot instance is preempted:

Before LMCache:
```
Spot preempted -> GPU KV cache destroyed -> next request does full prefill
```

With LMCache:
```
Spot preempted -> GPU cache destroyed BUT KV is still in Redis/S3
                -> next request goes to serverless
                -> serverless retrieves KV from Redis, skips prefill
                -> near-zero TTFT penalty from preemption
```

The KV cache outlives the instance. Preemption becomes a non-event for users.

**4. Cold start mitigation on serverless (YES)**

Serverless cold starts mean no cached state. With LMCache + shared backend:
```
First request on cold serverless container:
  -> LMCache checks Redis for matching prefix
  -> if system prompt / common prefix cached: retrieve, skip prefill
  -> only compute the novel suffix
```

For workloads with shared system prompts (all requests share the same instruction
prefix), this dramatically cuts cold-start TTFT.

**5. CPU offloading on every instance (YES, easy win)**

Even without shared remote backends, just enabling LMCache's CPU offloading
(`LMCACHE_LOCAL_CPU=True`) on every instance gives 7.4x TTFT improvement for
repeated prefixes on the same instance. Zero infrastructure needed. Pure env var
change.

### What's Still Not Feasible

**1. Disaggregated prefill/decode across providers (NO)**

Still requires low-latency KV transfer between prefill and decode workers. LMCache
supports disaggregated P/D via NIXL, but only within the same network (RDMA).
Cross-provider P/D disaggregation would add seconds of latency, defeating the purpose.

Within a multi-replica spot fleet on the same cloud, this COULD work if instances
are co-located.

**2. Real-time radix tree across all providers (IMPRACTICAL)**

LMCache's ZMQ KV events could theoretically feed a radix tree, but the latency
of cross-provider event propagation makes this less useful than just querying the
Controller's `/lookup` on demand.

**3. RunPod LMCache integration (NEEDS WORK)**

RunPod uses its own vLLM worker image. We'd need either:
- RunPod to add LMCache support to their image (feature request)
- Custom RunPod image with LMCache pre-installed
- OR: skip LMCache on RunPod, use it on Modal + Cloud Run + spot only

### Difficulty Matrix

| Capability | Difficulty | What's Needed |
|---|---|---|
| CPU offloading (per-instance) | **Trivial** | Add 3 env vars to each provider |
| Shared Redis backend | **Easy** | Deploy Redis, add REMOTE_URL env var |
| Shared S3 backend | **Easy** | Create bucket, add REMOTE_URL env var |
| KV-aware routing via /lookup | **Medium** | Deploy LMCache Controller, add lookup call to router |
| Spot preemption KV survival | **Medium** | Redis/S3 backend + router drain logic |
| Cold start mitigation | **Free** | Falls out of shared backend automatically |
| RunPod LMCache support | **Hard** | Custom image or RunPod feature request |
| P/D disaggregation (within spot) | **Hard** | NIXL setup, co-located instances, proxy |

---

## 7. Concrete Architecture Vision (Revised)

### Phase 1: CPU Offloading + Shared Cache (Easy, High Impact)

**What**: Enable LMCache on all providers with CPU offloading + a shared Redis.

**Changes to tuna**:
- Deploy a Redis instance (managed Redis on any cloud, or self-hosted)
- Add LMCache env vars to each provider's `plan()` method:
  ```python
  env["LMCACHE_CHUNK_SIZE"] = "256"
  env["LMCACHE_LOCAL_CPU"] = "True"
  env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
  env["LMCACHE_REMOTE_URL"] = f"redis://{redis_host}:6379"
  env["LMCACHE_REMOTE_SERDE"] = "cachegen"
  ```
- Add `--kv-transfer-config` to vLLM args (Modal, Cloud Run, SkyPilot)
- Optional: add `--enable-prefix-caching` to vLLM args

**Impact**: Every instance shares KV cache. Multi-turn conversations, shared system
prompts, and RAG contexts are cached once and reused everywhere. Spot preemption
no longer destroys cached context.

**No router changes needed** — this is purely a provider-level config change.

### Phase 2: Queue-Aware Overflow + Prefix Routing (Medium)

**What**: Add load-awareness and cache-awareness to the router.

**Changes to `meta_lb.py`**:
- Track in-flight requests per backend
- Track rolling p95 latency per backend
- Periodically scrape spot's `/metrics` for `vllm:num_requests_waiting`
- On each request, hash the prompt prefix and check LMCache Controller `/lookup`
- Route to the backend with the deepest cache hit, subject to load constraints

**Decision logic**:
```
1. Query LMCache Controller /lookup for prefix match per backend
2. Among backends with cache hits:
     - Filter to those with queue_depth < threshold AND p95 < SLO
     - Pick the deepest match
3. If no cache hits or all matched backends overloaded:
     - If spot healthy and under capacity: route to spot
     - Else: route to serverless
4. If nothing available: 503
```

**Impact**: Requests go to the backend that already has their KV cached AND has
capacity. Bursty traffic spills to serverless. Spot saturation is detected and
handled before users feel it.

### Phase 3: Multi-Replica Spot Fleet + KV Events (Long-term)

**What**: Scale to multiple spot replicas with intra-fleet KV optimization.

**Additional infrastructure**:
- LMCache Controller deployed alongside the router
- Each spot replica connects to the Controller with P2P enabled
- ZMQ KV events flow from spot replicas to Controller
- Controller maintains full prefix index for the spot fleet

**Within-fleet routing** (Dynamo-style):
```
cost = overlap_weight * new_prefill_blocks + active_decode_blocks
route to lowest-cost spot replica
```

**Cross-tier routing** (Phase 2 logic):
```
If all spot replicas saturated: overflow to serverless
Serverless can still retrieve KV from Redis (Phase 1 shared cache)
```

### Handling Spot Preemption (With LMCache)

```
On preemption signal (grace period start):
  1. Mark spot backend as "draining" in router
  2. Stop routing new requests to it
  3. Let in-flight requests complete (within grace period)
  4. KV cache is ALREADY in Redis — nothing is lost
  5. Remove affinity entries pointing to this backend
  6. Route everything to serverless
     -> serverless retrieves KV from Redis, minimal TTFT penalty

On new spot instance ready:
  1. Health check passes -> mark as "ready"
  2. LMCache on new instance auto-connects to same Redis
  3. New instance immediately benefits from cached KV in Redis
  4. No cold-cache penalty — the new spot instance starts warm
```

This is fundamentally different from the pre-LMCache world where spot preemption
meant starting from zero.

---

## 8. Open Questions

1. **Redis sizing and cost**: How much Redis memory is needed? KV cache for 7B
   model at 4K context is ~250MB compressed. At 100 concurrent conversations,
   that's ~25GB. Managed Redis at that scale is ~$200/month. Worth it? Compare
   against GPU compute savings from avoided prefills.

2. **S3 vs Redis trade-offs**: S3 is cheaper for storage but higher latency
   (~50-100ms per GET). Redis is faster (~1-5ms) but more expensive. For long
   contexts where prefill takes seconds, even S3 latency is acceptable. For short
   contexts, Redis is needed to beat recomputation.

3. **LMCache chunk size tuning**: Default is 256 tokens. Smaller chunks = more
   granular sharing but more overhead. Larger chunks = less overhead but less
   sharing. Needs benchmarking with tuna's actual workloads.

4. **RunPod integration path**: Options are (a) custom RunPod image with LMCache,
   (b) request RunPod add LMCache to their image, or (c) skip RunPod for
   LMCache features. Option (a) is the most realistic near-term.

5. **CacheBlend for RAG**: LMCache's CacheBlend enables non-prefix reuse. For
   RAG workloads where different documents are inserted into prompts at different
   positions, this could share KV cache across documents. Worth exploring.

6. **LMCache Controller deployment**: Where does it run? On the router VM?
   Separate container? The Controller needs to be reachable by all vLLM instances
   across providers. A lightweight VM or managed container would work.

7. **TTL and eviction policy**: How long should KV cache live in Redis? LMCache
   supports LRU/MRU/LFU/FIFO. For multi-turn conversations, cache should survive
   at least the conversation duration. For shared system prompts, cache should be
   effectively permanent.

8. **Multi-model support**: If tuna ever supports routing between different
   model sizes (e.g., small model for easy queries, large for hard ones), RouteLLM's
   cost-quality routing becomes relevant. LMCache Controller tracks per-model
   contexts, so this naturally extends.

9. **SkyPilot preemption signal latency**: How quickly does SkyPilot detect and
   propagate spot preemption signals? This determines whether we can drain
   gracefully or if requests will fail. With LMCache, even ungraceful preemption
   is survivable (KV is in Redis), but graceful drain is still better for
   in-flight requests.

---

## References

### Primary Systems
- **LMCache**: https://github.com/LMCache/LMCache / https://docs.lmcache.ai
- **Dynamo**: https://github.com/ai-dynamo/dynamo
- **AIBrix**: https://github.com/vllm-project/aibrix
- **SpotServe**: https://arxiv.org/abs/2311.15566 (ASPLOS 2024)
- **Splitwise**: https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/ (ISCA 2024)
- **DistServe**: https://arxiv.org/abs/2401.09670 (OSDI 2024)
- **Mooncake**: https://arxiv.org/abs/2407.00079

### Prefix-Aware Routing
- **SGLang RadixAttention**: https://arxiv.org/abs/2312.07104
- **Preble**: https://arxiv.org/abs/2407.00023
- **MemServe**: https://arxiv.org/abs/2406.17565

### Scheduling
- **Sarathi-Serve**: https://arxiv.org/abs/2403.02310
- **FastServe**: https://arxiv.org/abs/2305.05920
- **Andes**: https://arxiv.org/abs/2404.16283
- **Llumnix**: https://arxiv.org/abs/2406.03243

### KV Cache
- **CacheGen**: https://arxiv.org/abs/2310.07240 (SIGCOMM 2024)
- **Infinite-LLM**: https://arxiv.org/abs/2401.02669

### Cost-Aware
- **RouteLLM**: https://arxiv.org/abs/2406.18665
- **Helix**: https://arxiv.org/abs/2406.01566
- **ServerlessLLM**: https://www.usenix.org/conference/osdi24/presentation/fu
