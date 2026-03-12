#!/bin/bash
# replica_watcher.sh — pushes spot replica count to the router
# Args: SERVICE_NAME ROUTER_URL API_KEY POLL_INTERVAL
SERVICE_NAME="$1"
ROUTER_URL="$2"
API_KEY="$3"
POLL_INTERVAL="${4:-30}"

# Find the sky CLI binary (skypilot-runtime venv or system PATH)
SKY_BIN=""
for candidate in \
    "$HOME/skypilot-runtime/bin/sky" \
    "$(which sky 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        SKY_BIN="$candidate"
        break
    fi
done

if [ -z "$SKY_BIN" ]; then
    echo "ERROR: Could not find 'sky' CLI binary" >&2
    exit 1
fi
echo "Using sky CLI: $SKY_BIN" >&2

while true; do
    # Run 'sky serve status SERVICE_NAME' and count READY replicas
    OUTPUT=$("$SKY_BIN" serve status "$SERVICE_NAME" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$OUTPUT" ]; then
        # Count lines in 'Service Replicas' section with status READY
        REPLICAS=$(echo "$OUTPUT" | awk -v svc="$SERVICE_NAME" '
            /Service Replicas/ { in_replicas=1; next }
            in_replicas && $0 ~ svc && /READY/ { ready++ }
            END { print ready+0 }
        ')
    else
        REPLICAS="-1"
    fi

    if [ "$REPLICAS" != "-1" ]; then
        curl -s -X POST "$ROUTER_URL/router/spot-replicas" \
            -H "Content-Type: application/json" \
            -H "x-api-key: $API_KEY" \
            -d "{\"replicas\": $REPLICAS}" > /dev/null 2>&1
        echo "$(date -u '+%Y-%m-%d %H:%M:%S') Pushed replicas=$REPLICAS" >&2
    else
        echo "$(date -u '+%Y-%m-%d %H:%M:%S') sky serve status failed, skipping" >&2
    fi

    sleep "$POLL_INTERVAL"
done
