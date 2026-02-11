# Router Overflow Strategies

## Current State

The router (`meta_lb.py`) uses binary routing: spot is either "ready" or "not ready"
based on a 1s health probe. When spot is ready, 100% of traffic goes there. No
awareness of saturation, queue depth, or latency.

## Proposed Strategies (simple to complex)

### 1. Concurrent In-Flight Tracking (simplest)

Track how many requests are currently in-flight to spot. When count exceeds a
threshold (e.g. spot's concurrency setting), spill new requests to serverless.

- Pros: No external dependencies, pure router-level logic
- Cons: Doesn't account for request complexity (short vs long generation)

### 2. Latency-Based Spillover

Track rolling average/p95 of spot response times. When latency crosses a threshold,
start routing a percentage of traffic to serverless. Shift back as latency recovers.

- Pros: Directly measures what users care about
- Cons: Reactive (latency must degrade before spillover kicks in)

### 3. vLLM Queue Depth (most accurate)

vLLM exposes `/metrics` (Prometheus) with `vllm:num_requests_waiting`. Health probe
checks this — if queue is deep, router knows spot is saturated before clients feel it.

- Pros: Proactive, catches saturation before latency degrades
- Cons: Requires parsing Prometheus metrics, vLLM-specific

### 4. Hybrid: In-Flight + Timeout Fallback

Track in-flight requests. If a spot request doesn't respond within N seconds (shorter
than the 210s upstream timeout), cancel it and retry on serverless. Hard bound on
user-facing latency.

- Pros: Guarantees latency SLA
- Cons: Wasted compute on cancelled spot requests

### 5. LMCache Controller /lookup (best of all worlds)

Deploy LMCache on all vLLM instances with a shared Redis backend. The LMCache
Controller's `/lookup` API tells the router which backend has the deepest KV cache
hit for any given prompt. Route to the backend with the best hit AND lowest load.

- Pros: Cache-aware, works across providers, KV survives spot preemption
- Cons: Requires Redis infra and LMCache Controller deployment
- See: `docs/kv-aware-routing-research.md` sections 5-7 for full architecture

## Future: KV-Aware Routing

See `docs/kv-aware-routing-research.md` for deep research on Dynamo, AIBrix,
LMCache, and feasibility for hybrid spot/serverless architectures.
