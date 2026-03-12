"""Generate aiperf-compatible mooncake_trace JSONL files with traffic patterns.

Creates timestamped request traces with Poisson-distributed arrivals that
include zero-traffic gaps for spot scale-down testing.  Feed the output to
aiperf with ``--input-file <trace> --custom-dataset-type mooncake-trace
--fixed-schedule --fixed-schedule-auto-offset``.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Traffic profiles
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """One segment of a traffic pattern."""
    name: str
    start_pct: float   # 0.0–1.0
    end_pct: float
    rps: float          # requests per second (0 = zero-traffic gap)


DAY_CYCLE_PHASES: list[Phase] = [
    # Ramp up
    Phase("ramp",        0.00, 0.10, 0.5),   # starts slow, avg 0.5 rps
    # Morning peak
    Phase("peak",        0.10, 0.25, 3.0),
    # Wind down
    Phase("wind-down-1", 0.25, 0.30, 1.0),
    # GAP 1 — zero traffic
    Phase("gap-1",       0.30, 0.35, 0.0),
    # Recovery
    Phase("recovery-1",  0.35, 0.37, 1.0),
    # Steady state
    Phase("steady-1",    0.37, 0.50, 2.0),
    # Afternoon spike
    Phase("spike",       0.50, 0.58, 3.0),
    # Wind down
    Phase("wind-down-2", 0.58, 0.63, 1.0),
    # GAP 2 — zero traffic
    Phase("gap-2",       0.63, 0.68, 0.0),
    # Recovery
    Phase("recovery-2",  0.68, 0.70, 1.0),
    # Evening steady
    Phase("steady-2",    0.70, 0.85, 2.0),
    # Wind down
    Phase("wind-down-3", 0.85, 0.90, 1.0),
    # GAP 3 — zero traffic
    Phase("gap-3",       0.90, 0.95, 0.0),
    # Final trickle
    Phase("trickle",     0.95, 1.00, 0.3),
]

FLAT_PHASES: list[Phase] = [
    Phase("flat", 0.0, 1.0, 3.0),
]

SPIKE_PHASES: list[Phase] = [
    Phase("baseline",  0.00, 0.18, 0.3),
    Phase("spike-1",   0.18, 0.22, 5.0),
    Phase("baseline",  0.22, 0.38, 0.3),
    Phase("spike-2",   0.38, 0.42, 5.0),
    Phase("baseline",  0.42, 0.58, 0.3),
    Phase("spike-3",   0.58, 0.62, 5.0),
    Phase("baseline",  0.62, 0.78, 0.3),
    Phase("spike-4",   0.78, 0.82, 5.0),
    Phase("baseline",  0.82, 1.00, 0.3),
]

PROFILES: dict[str, list[Phase]] = {
    "day-cycle": DAY_CYCLE_PHASES,
    "flat": FLAT_PHASES,
    "spike": SPIKE_PHASES,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_trace(
    duration_s: float,
    profile: str = "day-cycle",
    isl: int = 200,
    osl: int = 100,
    isl_stddev: int = 50,
    osl_stddev: int = 30,
    seed: int | None = None,
) -> list[dict]:
    """Generate a list of mooncake_trace entries with timestamps.

    Returns a list of dicts, each with:
      - timestamp (int): milliseconds from start
      - input_length (int): token count for input
      - output_length (int): token count for output

    Zero-traffic gaps have no entries — aiperf won't send requests
    during those periods, letting the autoscaler scale to zero.
    """
    rng = random.Random(seed)
    phases = PROFILES.get(profile, DAY_CYCLE_PHASES)
    entries: list[dict] = []

    for phase in phases:
        if phase.rps <= 0:
            # Zero-traffic gap — no entries
            continue

        phase_start_s = phase.start_pct * duration_s
        phase_end_s = phase.end_pct * duration_s
        phase_dur_s = phase_end_s - phase_start_s

        if phase_dur_s <= 0:
            continue

        # Generate Poisson-distributed inter-arrival times
        mean_interval = 1.0 / phase.rps
        t = phase_start_s

        while t < phase_end_s:
            # Exponential inter-arrival (Poisson process)
            interval = rng.expovariate(1.0 / mean_interval)
            t += interval

            if t >= phase_end_s:
                break

            input_len = max(10, int(rng.gauss(isl, isl_stddev)))
            output_len = max(5, int(rng.gauss(osl, osl_stddev)))

            entries.append({
                "timestamp": int(t * 1000),  # milliseconds
                "input_length": input_len,
                "output_length": output_len,
            })

    # Sort by timestamp (should be mostly sorted, but phases might overlap at edges)
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def write_trace(
    path: str,
    duration_s: float,
    profile: str = "day-cycle",
    isl: int = 200,
    osl: int = 100,
    seed: int | None = None,
) -> int:
    """Generate trace and write to JSONL file. Returns number of entries."""
    entries = generate_trace(
        duration_s=duration_s,
        profile=profile,
        isl=isl,
        osl=osl,
        seed=seed,
    )
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return len(entries)


def print_trace_summary(entries: list[dict], duration_s: float, profile: str) -> None:
    """Print a human-readable summary of the generated trace."""
    if not entries:
        print("Empty trace.")
        return

    phases = PROFILES.get(profile, DAY_CYCLE_PHASES)
    total = len(entries)
    print(f"Trace: {total} requests over {duration_s/3600:.1f}h ({profile} profile)")
    print()

    for phase in phases:
        start_ms = int(phase.start_pct * duration_s * 1000)
        end_ms = int(phase.end_pct * duration_s * 1000)
        count = sum(1 for e in entries if start_ms <= e["timestamp"] < end_ms)
        dur_m = (phase.end_pct - phase.start_pct) * duration_s / 60
        actual_rps = count / (dur_m * 60) if dur_m > 0 else 0
        gap = " (ZERO TRAFFIC)" if phase.rps == 0 else ""
        print(f"  {phase.name:15s}  {dur_m:5.1f}m  target={phase.rps:.1f} rps  actual={actual_rps:.1f} rps  n={count}{gap}")

    print(f"\n  Total: {total} requests")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 2 * 3600
    profile = sys.argv[2] if len(sys.argv) > 2 else "day-cycle"
    output = sys.argv[3] if len(sys.argv) > 3 else "/tmp/tuna_trace.jsonl"

    entries = generate_trace(duration_s=duration, profile=profile, seed=42)
    print_trace_summary(entries, duration, profile)

    with open(output, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"\nWritten to {output}")
