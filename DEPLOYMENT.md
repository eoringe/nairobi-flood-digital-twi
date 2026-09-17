# Deploying to Railway

The dashboard ships as one Docker image. Railway builds it from the repository's
`Dockerfile` using the settings in `railway.json`.

---

## What runs in the container

On start, gunicorn serves the Dash app on Railway's `PORT`. A background thread
then warms everything up, which takes about a minute:

1. Loads the U-Net, its calibration and the WorldPop population grid.
2. Builds the road graph for routing and the location search index.
3. Fetches the live Open-Meteo forecast and prepares the three storm replays.
4. Pre-computes the What-if rainfall scenarios.

The app serves pages throughout. Anything a user requests before warm-up
finishes is computed on demand.

`GET /healthz` returns `{"status": "ok", "warmup_complete": ...}`. Railway uses
it to decide when a deploy is live.

Nothing else needs to run. Data collection and model training are offline steps
(Colab and `src/ingestion/`), and their outputs are committed and built into the
image.

---

## Requirements

| | |
|---|---|
| Plan | **Hobby or higher.** The app uses ~650 MB of RAM at rest and more while rendering a large flood, above the free trial's 0.5 GB. |
| Replicas | **1.** The model and caches live in process memory (`railway.json` sets this). |
| Outbound internet | Open-Meteo (forecasts) and Photon (search fallback). Allowed by default on Railway. |

---

## First deploy

### Option A: from GitHub (recommended)

1. Merge the pull requests into the branch you will deploy (normally `main`).
   Railway builds only what is committed on that branch.
2. In Railway, choose **New Project → Deploy from GitHub repo**, and select this
   repository and branch. Railway detects `railway.json` and builds the Dockerfile.
3. Under **Settings → Networking**, choose **Generate Domain** to get a public
   HTTPS URL.
4. The first build takes about 6–10 minutes (PyTorch). Later builds reuse the
   dependency layer unless `requirements-dashboard.txt` changes.

Every push to the deployed branch then redeploys automatically.

### Option B: from your machine with the Railway CLI

```
railway login
railway init            # or: railway link, for an existing project
railway up              # uploads the folder and builds it on Railway
railway domain          # generate a public URL
```

`railway up` skips files ignored by `.gitignore`. The runtime files the image
needs are whitelisted there.

---

## Keeping scenario history across deploys (optional)

Without a volume the app works fully. Scenario history and the cached live
forecast then reset on each deploy.

To keep them:

1. In the service, choose **New → Volume** and set the mount path to `/app/state`.
2. Under **Variables**, add `RAILWAY_RUN_UID=0`.

Railway mounts volumes as root, and the image runs as a non-root user, so this
variable is Railway's documented way to let the app write to the volume. Without
it, the app logs a warning and carries on without saving.

---

## Environment variables

All are optional; the image sets sensible defaults.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | set by Railway | Port gunicorn listens on |
| `WEB_THREADS` | `8` | Request threads in the single worker |
| `OMP_NUM_THREADS` | `4` | PyTorch CPU threads; match your plan's vCPUs |
| `TWIN_STATE_DIR` | `/app/state` | Where scenario history and the forecast cache are written |
| `LOG_LEVEL` | `info` | gunicorn log level |
| `RAILWAY_RUN_UID` | unset | Set to `0` only when a volume is attached (see above) |

---

## Checking a deploy

- **Health:** `https://<your-domain>/healthz` should show `"warmup_complete": true`
  about a minute after start.
- **Logs:** in the Railway dashboard or with `railway logs`. Look for
  `Warm-up complete: road network, outlooks and what-if scenarios cached.`

| Symptom | Likely cause |
|---|---|
| Deploy fails its healthcheck | The app crashed on start. Check the logs for a traceback. |
| Service restarts with no traceback | Out of memory. Use a plan with at least 2 GB. |
| "Forecast unavailable" | Open-Meteo is unreachable from Railway. The app retries every minute. |
| Replays slow the first time | Their rainfall is downloaded once, unless `data/processed/replay_rainfall.json` is committed on the deployed branch. |
| `Could not write /app/state/...` warning | A volume is attached without `RAILWAY_RUN_UID=0`. |
