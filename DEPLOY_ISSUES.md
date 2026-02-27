# Deployment Issues Found During BYOC GCP Testing
*Discovered 2026-02-27 — not currently in README*

---

## 1. Stale `gcloud` symlink after snap auto-update

**What happened:** `/usr/local/bin/gcloud` was symlinked to `/snap/google-cloud-cli/273/bin/gcloud` — but snap auto-updated and the versioned path no longer existed. Tuna preflight reported "gcloud CLI not found" even though gcloud was installed.

**Fix:** Always symlink to `current` not a versioned path:
```bash
sudo ln -sf /snap/google-cloud-cli/current/bin/gcloud /usr/local/bin/gcloud
```

**README gap:** Should mention that if using snap-installed gcloud, symlink to `current/` not a versioned directory.

---

## 2. `google-cloud-compute` not installed despite `skypilot[gcp]` being present

**What happened:** Tuna's GCP spot dependency check looks for `google.cloud.compute_v1`. Even though `skypilot[gcp]==0.11.1` was installed in the venv, `google-cloud-compute` was not — causing the error `"GCP spot requires additional dependencies"` even with SkyPilot GCP available.

**Fix:**
```bash
pip install google-cloud-compute
```

**README gap:** The error message says `pip install tandemn-tuna[gcp-spot]` but that alone doesn't pull `google-cloud-compute` in some environments. Should explicitly list it as a dependency or fix the `[gcp-spot]` extra in `pyproject.toml`.

---

## 3. Application Default Credentials (ADC) required separately from `gcloud auth login`

**What happened:** `gcloud auth login` sets credentials for the `gcloud` CLI, but the Cloud Run Python SDK (`google-cloud-run`) uses **Application Default Credentials (ADC)**, which are a completely separate credential set. Deploy fails with:
```
google.auth.exceptions.RefreshError: Reauthentication is needed.
Please run `gcloud auth application-default login` to reauthenticate.
```

**Fix:**
```bash
gcloud auth application-default login
```
Then re-run `tuna deploy`.

**README gap:** The Prerequisites section should add:
```bash
gcloud auth application-default login
```
as a required step alongside `gcloud auth login`. This is a very common gotcha for new GCP users.

---

## 4. Cloud Run L4 GPU quota is a producer override — cannot self-serve increase

**What happened:** Cloud Run L4 GPU quota is set by Google as a `producerOverride`, not a `consumerOverride`. This means:
- The "Edit" button in GCP Console is greyed out
- The Cloud Quotas API auto-denies increase requests for new/low-spend projects
- Filing a support ticket requires a paid support plan (Basic = $0 plan does NOT work)

**What you actually get:** Projects are auto-granted **3 L4 GPUs** per region when you first deploy. To get more you need to either:
1. Have billing history on the project (spend money first, then request)
2. Pay for Developer support ($29/mo) and file a ticket

**README gap:** The BYOC section should warn: *"Cloud Run GPU quota is capped at 3 per region by default. Increasing it requires a paid GCP support plan or prior billing history on the project."*

---

## 5. Cloud Run GPU only available in specific regions

**What happened:** Attempting to deploy in `us-central1` fails silently because Cloud Run L4 GPU quota is only available in `us-east4` and `europe-west1` for this project (auto-granted on first use). Other regions have 0 quota with no explanation.

**Fix:** Pass `--gcp-region us-east4` (or `europe-west1`).

**README gap:** Should document the supported Cloud Run GPU regions and note that `--gcp-region` defaults to `us-central1` which may have 0 GPU quota.

---

## 6. `GOOGLE_CLOUD_PROJECT` env var needed alongside `--gcp-project` flag

**What happened:** The `--gcp-project` flag sets `GOOGLE_CLOUD_PROJECT` env var internally, but if deploying programmatically (not via CLI), this needs to be set manually. The Cloud Run provider reads from env var, not from DeployRequest directly.

**This is minor** — CLI users won't hit it, but worth noting for anyone using Tuna as a library.
