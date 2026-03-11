#!/bin/bash
# replica_watcher.sh — pushes spot replica count to the router
# Args: SERVICE_NAME ROUTER_URL API_KEY POLL_INTERVAL
SERVICE_NAME="$1"
ROUTER_URL="$2"
API_KEY="$3"
POLL_INTERVAL="${4:-30}"

# Find the Python with SkyPilot installed (skypilot-runtime venv or system)
SKY_PYTHON=""
for candidate in \
    "$HOME/skypilot-runtime/bin/python" \
    "$(which python3 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        if "$candidate" -c "import sky" 2>/dev/null; then
            SKY_PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$SKY_PYTHON" ]; then
    echo "ERROR: Could not find Python with SkyPilot installed" >&2
    exit 1
fi
echo "Using Python: $SKY_PYTHON" >&2

while true; do
    # Parse 'sky serve status' CLI output — more stable than internal Python API
    REPLICAS=$("$SKY_PYTHON" -c "
import subprocess, re
try:
    result = subprocess.run(
        ['$SKY_PYTHON', '-m', 'sky.cli', 'serve', 'status', '$SERVICE_NAME'],
        capture_output=True, text=True, timeout=30,
    )
    # Count lines in 'Service Replicas' section with status READY
    lines = result.stdout.strip().split('\n')
    ready = 0
    in_replicas = False
    for line in lines:
        if 'Service Replicas' in line:
            in_replicas = True
            continue
        if in_replicas and '$SERVICE_NAME' in line and 'READY' in line:
            ready += 1
    print(ready)
except:
    print(-1)
" 2>/dev/null)

    if [ "$REPLICAS" != "-1" ]; then
        curl -s -X POST "$ROUTER_URL/router/spot-replicas" \
            -H "Content-Type: application/json" \
            -H "x-api-key: $API_KEY" \
            -d "{\"replicas\": $REPLICAS}" > /dev/null 2>&1
    fi

    sleep "$POLL_INTERVAL"
done
