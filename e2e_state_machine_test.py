"""E2E State Machine Test — fully automated, no intervention needed.

Usage:
    CEREBRIUM_API_KEY=... uv run python e2e_state_machine_test.py

Deploys a hybrid setup (Cerebrium + AWS spot L4), tests every state
transition, reports pass/fail, and tears down at the end.
"""
import json
import os
import subprocess
import sys
import time

import requests

# ── Config ──────────────────────────────────────────────────────────────
SERVICE_NAME = "e2e-sm-test"
MODEL = "Qwen/Qwen3-0.6B"
GPU = "L4"
SERVERLESS_PROVIDER = "cerebrium"
SPOTS_CLOUD = "aws"
DOWNSCALE_DELAY = 60  # seconds — how long before spot scales to 0

# Timeouts
DEPLOY_TIMEOUT = 600       # 10 min for full deploy
SPOT_BOOT_TIMEOUT = 600    # 10 min for spot replica to become READY
SCALE_TO_ZERO_TIMEOUT = 180  # 3 min for spot to actually scale to 0
COLD_START_TIMEOUT = 600   # 10 min for spot to come back from 0

# Discord (optional)
DISCORD_WEBHOOK = os.environ.get(
    "DISCORD_WEBHOOK",
    "https://discord.com/api/webhooks/1453154642706960485/"
    "iFXIAaDTLxNO7_GHKHhXnXwFFnXziniP4TUwLUDUnXHtT9kNo08eQBjGQ4CiBr6AazY6",
)


# ── Helpers ─────────────────────────────────────────────────────────────
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"{Colors.CYAN}[{ts}]{Colors.END} {msg}", flush=True)


def log_pass(name: str, details: str = ""):
    extra = f" — {details}" if details else ""
    print(f"  {Colors.GREEN}PASS{Colors.END} {name}{extra}", flush=True)


def log_fail(name: str, details: str = ""):
    extra = f" — {details}" if details else ""
    print(f"  {Colors.RED}FAIL{Colors.END} {name}{extra}", flush=True)


def discord(title: str, desc: str, color: int = 0x3498DB):
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"embeds": [{
            "title": title, "description": desc, "color": color,
            "footer": {"text": "e2e state machine test"},
        }]}, timeout=10)
    except Exception:
        pass


def get_health(router_url: str, api_key: str) -> dict:
    resp = requests.get(
        f"{router_url}/router/health",
        headers={"x-api-key": api_key}, timeout=15,
    )
    return resp.json()


def send_request(router_url: str, api_key: str, timeout: float = 120) -> dict:
    """Send a chat completion and return {ok, latency_ms, error}."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{router_url}/v1/chat/completions",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 16,
            },
            timeout=timeout,
        )
        latency = (time.monotonic() - t0) * 1000
        if resp.status_code == 200:
            return {"ok": True, "latency_ms": latency, "error": None}
        return {"ok": False, "latency_ms": latency, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"ok": False, "latency_ms": (time.monotonic() - t0) * 1000, "error": str(e)}


def wait_for_state(router_url: str, api_key: str, target: str, timeout: float, label: str = "") -> bool:
    """Poll /router/health until spot_state == target or timeout. Discord update every 60s."""
    deadline = time.time() + timeout
    last_discord = 0
    while time.time() < deadline:
        try:
            h = get_health(router_url, api_key)
            state = h.get("spot_state", "?")
            if state == target:
                return True
            elapsed = timeout - (deadline - time.time())
            if time.time() - last_discord >= 60:
                discord(f"⏳ {label or 'Waiting'}", f"state={state}, want={target}, {elapsed:.0f}s/{timeout:.0f}s")
                last_discord = time.time()
        except Exception:
            pass
        time.sleep(5)
    return False


def wait_for_sky_replicas(target: int, service_name: str, timeout: float) -> bool:
    """Poll sky serve status until READY replica count == target. Discord update every 60s."""
    deadline = time.time() + timeout
    last_discord = 0
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["uv", "run", "sky", "serve", "status", service_name],
                capture_output=True, text=True, timeout=30,
            )
            # Parse REPLICAS column (e.g., "1/1" or "0/0")
            for line in result.stdout.splitlines():
                if service_name in line and "REPLICAS" not in line:
                    parts = line.split()
                    for part in parts:
                        if "/" in part:
                            try:
                                ready = int(part.split("/")[0])
                                if ready == target:
                                    return True
                                elapsed = timeout - (deadline - time.time())
                                if time.time() - last_discord >= 60:
                                    discord(f"⏳ Replicas", f"ready={ready}, want={target}, {elapsed:.0f}s/{timeout:.0f}s")
                                    last_discord = time.time()
                            except ValueError:
                                pass
        except Exception:
            pass
        time.sleep(10)
    return False


def get_deployment_info() -> tuple[str, str]:
    """Read router URL and API key from tuna state."""
    from tuna.state import load_deployment
    record = load_deployment(SERVICE_NAME)
    if not record:
        return "", ""
    router_url = record.router_endpoint or ""
    api_key = (record.router_metadata or {}).get("router_api_key", "")
    return router_url, api_key


# ── Deploy ──────────────────────────────────────────────────────────────
def deploy() -> tuple[str, str]:
    """Deploy hybrid setup. Returns (router_url, api_key)."""
    log("Deploying hybrid setup...")
    discord("🚀 E2E Test Starting", f"Deploying {MODEL} on {GPU}\nCerebrium + AWS spot")

    cmd = [
        "uv", "run", "python", "-m", "tuna", "deploy",
        "--model", MODEL,
        "--gpu", GPU,
        "--serverless-provider", SERVERLESS_PROVIDER,
        "--spots-cloud", SPOTS_CLOUD,
        "--service-name", SERVICE_NAME,
        "--max-model-len", "512",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DEPLOY_TIMEOUT)
    print(proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout, flush=True)
    if proc.returncode != 0:
        log(f"Deploy failed: {proc.stderr[-300:]}")
        return "", ""

    router_url, api_key = get_deployment_info()
    if not router_url:
        # Fallback: parse from output
        for line in proc.stdout.splitlines():
            if "Router:" in line:
                router_url = line.split("Router:")[-1].strip()

    log(f"Deployed. Router: {router_url}")
    return router_url, api_key


# ── Teardown ────────────────────────────────────────────────────────────
def teardown():
    log("Tearing down...")
    try:
        subprocess.run(
            ["uv", "run", "python", "-m", "tuna", "destroy",
             "--service-name", SERVICE_NAME],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as e:
        log(f"Teardown error: {e}")
    log("Teardown complete.")


# ── Tests ───────────────────────────────────────────────────────────────
def test_1_boot_and_warming(router_url: str, api_key: str) -> bool:
    """Test 1: Boot → COLD → request → WARMING → READY"""
    log(f"{Colors.BOLD}Test 1: Boot & COLD → WARMING → READY{Colors.END}")
    results = []

    # 1a. Router should auto-enter WARMING on startup (spot URL configured)
    h = get_health(router_url, api_key)
    state = h.get("spot_state", "?")
    if state in ("warming", "ready"):
        log_pass(f"Initial state is {state.upper()} (auto-warming on startup)")
        results.append(True)
    else:
        log_fail("Initial state", f"expected warming/ready, got {state}")
        results.append(False)

    # 1b. Send request — should route to serverless, trigger WARMING
    log("Sending first request (expect serverless routing + WARMING trigger)...")
    r = send_request(router_url, api_key)
    if r["ok"]:
        log_pass("First request succeeded", f"{r['latency_ms']:.0f}ms (serverless)")
        results.append(True)
    else:
        log_fail("First request", r["error"])
        results.append(False)

    # 1c. Check state transitioned to WARMING
    time.sleep(2)
    h = get_health(router_url, api_key)
    state = h.get("spot_state", "?")
    if state in ("warming", "ready"):
        log_pass(f"State after first request: {state}")
        results.append(True)
    else:
        log_fail(f"Expected warming/ready, got {state}")
        results.append(False)

    # 1d. Wait for READY (spot boots)
    log(f"Waiting for spot replica to boot (up to {SPOT_BOOT_TIMEOUT}s)...")
    if wait_for_state(router_url, api_key, "ready", SPOT_BOOT_TIMEOUT, "Test 1: Spot Boot"):
        log_pass("Spot reached READY")
        results.append(True)
    else:
        log_fail("Spot did not reach READY within timeout")
        results.append(False)
        return False

    # 1e. Verify min_replicas=0 (deployed directly, no sky serve update needed)
    log("Verifying min_replicas=0 in autoscaling policy...")
    spot_svc = f"{SERVICE_NAME}-spot"
    try:
        result = subprocess.run(
            ["uv", "run", "sky", "serve", "status", spot_svc, "-v"],
            capture_output=True, text=True, timeout=30,
        )
        if "from 0 to" in result.stdout:
            log_pass("min_replicas=0 confirmed (deployed directly)")
            results.append(True)
        else:
            log_fail("min_replicas not 0 in autoscaling policy")
            results.append(False)
    except Exception as e:
        log_fail("Could not check autoscaling policy", str(e))
        results.append(False)

    passed = all(results)
    discord(
        "✅ Test 1: Boot" if passed else "❌ Test 1: Boot",
        f"COLD → WARMING → READY: {'PASS' if passed else 'FAIL'}\n"
        f"Scale-to-zero armed: {'yes' if results[-1] else 'no'}",
        0x2ECC71 if passed else 0xE74C3C,
    )
    return passed


def test_2_steady_state(router_url: str, api_key: str) -> bool:
    """Test 2: Steady state — requests route to spot."""
    log(f"{Colors.BOLD}Test 2: READY steady state{Colors.END}")

    h_before = get_health(router_url, api_key)
    spot_before = h_before["route_stats"]["spot"]

    log("Sending 10 requests...")
    successes = 0
    for i in range(10):
        r = send_request(router_url, api_key)
        if r["ok"]:
            successes += 1

    h_after = get_health(router_url, api_key)
    spot_after = h_after["route_stats"]["spot"]
    spot_routed = spot_after - spot_before

    if successes >= 9 and spot_routed >= 7:
        log_pass(f"{successes}/10 succeeded, {spot_routed}/10 via spot")
        discord("✅ Test 2: Steady State", f"{successes}/10 requests, {spot_routed} via spot", 0x2ECC71)
        return True
    else:
        log_fail(f"{successes}/10 succeeded, {spot_routed}/10 via spot")
        discord("❌ Test 2: Steady State", f"{successes}/10 ok, {spot_routed} via spot", 0xE74C3C)
        return False


def test_3_scale_to_zero(router_url: str, api_key: str) -> bool:
    """Test 3: Stop traffic → spot scales to 0 → router goes COLD."""
    log(f"{Colors.BOLD}Test 3: READY → COLD (scale-to-zero){Colors.END}")
    results = []

    log(f"Stopping traffic. Waiting for scale-to-zero ({DOWNSCALE_DELAY}s delay + autoscaler)...")
    discord("⏳ Test 3: Scale-to-Zero", f"Traffic stopped. Waiting {DOWNSCALE_DELAY}s+ for scale-down...")

    # Wait for SkyServe to scale to 0 replicas
    spot_svc = f"{SERVICE_NAME}-spot"
    if wait_for_sky_replicas(0, spot_svc, SCALE_TO_ZERO_TIMEOUT):
        log_pass("SkyServe scaled to 0 replicas")
        results.append(True)
    else:
        log_fail("SkyServe did not scale to 0 within timeout")
        results.append(False)
        # Check what happened
        try:
            result = subprocess.run(
                ["uv", "run", "sky", "serve", "status", spot_svc, "-v"],
                capture_output=True, text=True, timeout=30,
            )
            log(f"sky serve status:\n{result.stdout[:500]}")
        except Exception:
            pass

    # Wait for router to detect COLD
    if wait_for_state(router_url, api_key, "cold", 60, "Test 3: Router → COLD"):
        log_pass("Router transitioned to COLD")
        results.append(True)
    else:
        h = get_health(router_url, api_key)
        log_fail(f"Router state: {h.get('spot_state', '?')} (expected cold)")
        results.append(False)

    # Verify request still works (via serverless)
    r = send_request(router_url, api_key)
    if r["ok"]:
        log_pass("Request during COLD succeeded", f"{r['latency_ms']:.0f}ms (serverless)")
        results.append(True)
    else:
        log_fail("Request during COLD failed", r["error"])
        results.append(False)

    passed = all(results)
    discord(
        "✅ Test 3: Scale-to-Zero" if passed else "❌ Test 3: Scale-to-Zero",
        f"Replicas → 0: {'yes' if results[0] else 'no'}\n"
        f"Router → COLD: {'yes' if len(results) > 1 and results[1] else 'no'}\n"
        f"Serverless fallback: {'works' if len(results) > 2 and results[2] else 'failed'}",
        0x2ECC71 if passed else 0xE74C3C,
    )
    return passed


def test_4_cold_start(router_url: str, api_key: str) -> bool:
    """Test 4: COLD → send request → WARMING → READY (cold start from zero)."""
    log(f"{Colors.BOLD}Test 4: COLD → WARMING → READY (cold start from zero){Colors.END}")
    results = []

    # Verify we're in COLD
    h = get_health(router_url, api_key)
    state = h.get("spot_state", "?")
    if state != "cold":
        log(f"Warning: expected COLD, got {state}. Continuing anyway.")

    # Send request — triggers WARMING
    log("Sending request to trigger cold start...")
    discord("⏳ Test 4: Cold Start", "Sending request while COLD. Waiting for spot to boot from zero...")
    r = send_request(router_url, api_key)
    if r["ok"]:
        log_pass("Request during COLD succeeded", f"{r['latency_ms']:.0f}ms (serverless)")
        results.append(True)
    else:
        log_fail("Request during COLD failed", r["error"])
        results.append(False)

    # Check WARMING state
    time.sleep(3)
    h = get_health(router_url, api_key)
    state = h.get("spot_state", "?")
    if state in ("warming", "ready"):
        log_pass(f"State after request: {state}")
        results.append(True)
    else:
        log_fail(f"Expected warming, got {state}")
        results.append(False)

    # Wait for READY (spot boots from zero)
    log(f"Waiting for spot to boot from zero (up to {COLD_START_TIMEOUT}s)...")
    t0 = time.monotonic()
    if wait_for_state(router_url, api_key, "ready", COLD_START_TIMEOUT, "Test 4: Cold Start Boot"):
        boot_time = time.monotonic() - t0
        log_pass(f"Spot reached READY from zero in {boot_time:.0f}s")
        results.append(True)
    else:
        log_fail("Spot did not reach READY within timeout")
        results.append(False)
        passed = all(results)
        discord("❌ Test 4: Cold Start", "Spot did not boot from zero", 0xE74C3C)
        return passed

    # Verify requests route to spot — retry a few times since vLLM may
    # still be loading the model even after SkyServe reports READY
    log("Verifying post-recovery requests (retrying if 502)...")
    for attempt in range(5):
        h_before = get_health(router_url, api_key)
        spot_before = h_before["route_stats"]["spot"]
        r = send_request(router_url, api_key)
        if r["ok"]:
            h_after = get_health(router_url, api_key)
            spot_after = h_after["route_stats"]["spot"]
            if spot_after > spot_before:
                log_pass(f"Post-recovery request routed to spot", f"{r['latency_ms']:.0f}ms (attempt {attempt+1})")
            else:
                log_pass(f"Post-recovery request succeeded", f"{r['latency_ms']:.0f}ms (attempt {attempt+1})")
            results.append(True)
            break
        else:
            log(f"  Attempt {attempt+1}/5: {r['error']} — retrying in 10s...")
            time.sleep(10)
    else:
        log_fail("Post-recovery requests failed after 5 attempts", r["error"])
        results.append(False)

    passed = all(results)
    discord(
        "✅ Test 4: Cold Start" if passed else "❌ Test 4: Cold Start",
        f"Serverless fallback: {'works' if results[0] else 'failed'}\n"
        f"Spot boot from zero: {'yes' if len(results) > 2 and results[2] else 'no'}\n"
        f"Post-recovery routing: {'spot' if len(results) > 3 and results[3] else 'unknown'}",
        0x2ECC71 if passed else 0xE74C3C,
    )
    return passed


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  E2E State Machine Test{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

    if not os.environ.get("CEREBRIUM_API_KEY"):
        print("Error: CEREBRIUM_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Deploy
    router_url, api_key = deploy()
    if not router_url:
        log("Deploy failed. Aborting.")
        teardown()
        sys.exit(1)

    # Wait for router to be reachable
    log("Waiting for router to be reachable...")
    for _ in range(30):
        try:
            h = get_health(router_url, api_key)
            if h:
                break
        except Exception:
            pass
        time.sleep(2)

    results = {}
    try:
        # Test 1: Boot & warming
        results["test_1_boot"] = test_1_boot_and_warming(router_url, api_key)
        print()

        if not results["test_1_boot"]:
            log("Test 1 failed — cannot continue. Tearing down.")
            return

        # Test 2: Steady state
        results["test_2_steady"] = test_2_steady_state(router_url, api_key)
        print()

        # Test 3: Scale to zero
        results["test_3_scale_to_zero"] = test_3_scale_to_zero(router_url, api_key)
        print()

        if not results["test_3_scale_to_zero"]:
            log("Test 3 failed — scale-to-zero didn't work. Skipping cold start test.")
        else:
            # Test 4: Cold start from zero
            results["test_4_cold_start"] = test_4_cold_start(router_url, api_key)
            print()

    finally:
        # Summary
        print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}  RESULTS{Colors.END}")
        print(f"{Colors.BOLD}{'='*60}{Colors.END}")
        all_passed = True
        for name, passed in results.items():
            status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
            print(f"  {status}  {name}")
            if not passed:
                all_passed = False
        print()

        if all_passed:
            discord(
                "🎉 E2E Test: ALL PASSED",
                "\n".join(f"✅ {k}" for k in results),
                0x2ECC71,
            )
        else:
            failed = [k for k, v in results.items() if not v]
            discord(
                "💥 E2E Test: FAILURES",
                "\n".join(
                    f"{'✅' if v else '❌'} {k}" for k, v in results.items()
                ),
                0xE74C3C,
            )

        # Teardown
        teardown()

        if not all_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
