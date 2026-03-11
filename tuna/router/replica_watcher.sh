#!/bin/bash
# replica_watcher.sh — pushes spot replica count to the router
# Args: SERVICE_NAME ROUTER_URL API_KEY POLL_INTERVAL
SERVICE_NAME="$1"
ROUTER_URL="$2"
API_KEY="$3"
POLL_INTERVAL="${4:-30}"

while true; do
    REPLICAS=$(python3 -c "
from sky.serve.serve_utils import get_service_status
try:
    statuses = get_service_status('$SERVICE_NAME')
    if statuses:
        replicas = statuses[0].get('replica_info', [])
        ready = sum(1 for r in replicas if r.get('status_str') == 'READY')
        print(ready)
    else:
        print(0)
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
