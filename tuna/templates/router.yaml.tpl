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
  pip install flask requests gunicorn

run: |
  cd /app
  export SERVERLESS_BASE_URL="{serverless_url}"
  export SKYSERVE_BASE_URL="{spot_url}"
  gunicorn -w 1 -k gthread --threads 16 --timeout 300 \
    --bind 0.0.0.0:8080 meta_lb:app
