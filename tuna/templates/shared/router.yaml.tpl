name: {service_name}-router

resources:
  cpus: 2+
  memory: 4+
  use_spot: false
  ports: 8080
{region_block}

file_mounts:
  /app/meta_lb.py: {meta_lb_local_path}

setup: |
  pip install fastapi httpx 'uvicorn[standard]'

run: |
  cd /app
  export SERVERLESS_BASE_URL="{serverless_url}"
  export SKYSERVE_BASE_URL="{spot_url}"
  export API_KEY="{router_api_key}"
  export IDLE_TIMEOUT_SECONDS="{downscale_delay}"
  export WARMUP_POKE_INTERVAL_SECONDS="{upscale_delay}"
  uvicorn meta_lb:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 300
