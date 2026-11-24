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

## Keeping accounts and records across deploys: PostgreSQL (recommended)

The container's disk is replaced on every deploy. Without a database, accounts
and the roles given to them, the audit log, flood reports, prediction history
and saved scenarios are lost each time, and everyone signs in again as a new
citizen account.

1. In the project, choose **New → Database → Add PostgreSQL**.
2. In the dashboard service, under **Variables**, add
   `DATABASE_URL` = `${{Postgres.DATABASE_URL}}` (a reference to the database's
   private address; type `${{` and pick it from the list).
3. Also set `TWIN_SECRET_KEY` to a long random value, so sign-ins survive
   redeploys too.
4. Redeploy. The log shows `Records for 'accounts' are kept in PostgreSQL at ...`.

The tables are created on start. Each store has its own schema (`accounts`,
`operations`, `reports`, `scenarios`) in the one database. To bring across
records from SQLite files (for example from a machine that ran without a
database), run once with `DATABASE_URL` set:

```
python -m src.persistence.copy_to_postgres path/to/state-folder
```

It copies every row with its original id and can be re-run safely.

---

## Keeping the forecast cache across deploys (optional)

Without a volume the app works fully; the cached live forecast (and, without
`DATABASE_URL`, the SQLite records) then reset on each deploy.

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
| `DATABASE_URL` | unset | PostgreSQL for accounts, reports and history (see above). Unset: SQLite files in `TWIN_STATE_DIR` |
| `DB_POOL_SIZE` | `5` | Most connections each store keeps open to PostgreSQL |
| `LOG_LEVEL` | `info` | gunicorn log level |
| `RAILWAY_RUN_UID` | unset | Set to `0` only when a volume is attached (see above) |
| `TWIN_SECRET_KEY` | generated | Signs session cookies. Set a long random value so sign-ins survive redeploys |
| `TWIN_ADMIN_EMAIL` | unset | First administrator account, created only if none exists. They sign in with Google or an emailed link and set up an authenticator app |
| `TWIN_ADMIN_EMAILS` | unset | Google accounts that become administrators when they sign in |
| `TWIN_COOKIE_SECURE` | `0` | Set to `1` on Railway (HTTPS) so the session cookie is never sent over plain HTTP |
| `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_PROJECT_ID`, `FIREBASE_APP_ID` | unset | Sign-in with Google or an email link, the only ways to sign in. Also add the Railway domain under Firebase → Authentication → Settings → Authorized domains |

Accounts and the audit log are kept in PostgreSQL when `DATABASE_URL` is set.
Without it they are stored in `/app/state/accounts.db`, and every redeploy
without a volume starts with no users except the bootstrap administrator.

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
