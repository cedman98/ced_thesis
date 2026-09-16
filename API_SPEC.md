# API_SPEC.md — FastAPI layer over `forecast_service`

Status: **plan, not implemented.** Scope agreed in interview: trusted-network / localhost
now, web-accessible later; 1–5 concurrent users now; standalone but containerized;
robustness = *never silently stale*, *never a number without its interval*, *auditable*.

---

## 0. What I found in the existing code first

Three things that change the design:

**(a) The stale-forecast bug is real and live right now.**
`results/forecasts/brandenburg_state_forecast.parquet` covers
`2026-07-16 22:00Z → 2026-07-17 21:00Z`. Today is `2026-07-24`. `dashboard.py:187`
loads that parquet whenever the file *exists*, never checks its window, and
`dashboard.py:216` captions it `issued {now}`. The dashboard is currently presenting a
week-old forecast as today's. This is exactly the failure mode you ranked first, so the
freshness check is requirement #1 below — and it goes in `forecast_service.py`, not in
`api.py`, because `dashboard.py` is a second caller with the same bug. One guard, both
callers fixed.

**(b) `models/`, `data/`, `results/` are all gitignored.**
A Docker image built from the repo contains *no* models, *no* MaStR CSVs, *no* batch
forecast. This is not a packaging inconvenience, it's the central deployment constraint:
those three directories must be **volume mounts**, and the API must fail loudly at
startup if they're missing rather than 500 on first request.

**(c) `torch` + `pytorch-forecasting` + `jupyter` + `shap` are hard dependencies.**
The API path only touches `lightgbm`, `joblib`, `pandas`, `geopandas`, `requests`.
Installing `pyproject.toml` as-is into a container pulls the cu124 torch wheel: ~6 GB
image for endpoints that never import torch. Fix is a `pyproject.toml` edit, not code —
see §7.

**On your two follow-up questions:**

- *Web-accessible later?* Yes, and it costs nothing to prepare. Auth enters as a single
  FastAPI dependency on the router (§5). Building it now buys nothing; building it so it
  can be switched on with an env var is free.
- *Does GitHub matter?* Only for secrets, and only once auth exists. The API key lives in
  an env var, never in `conf/config.yaml` (which *is* committed). Since `data/` and
  `models/` are already gitignored, no asset or model data is exposed by the repo being
  public.

---

## 1. Shape

Two new files. Not a package.

```
api.py          # ~180 lines, all endpoints, mirrors dashboard.py's position at repo root
Dockerfile      # ~15 lines
```

No `src/api/` with `routers/`, `schemas/`, `services/`, `deps/`. Six endpoints do not
need a directory tree — `forecast_service.py` *is* the service layer, that's the whole
point of wrapping it.

Handlers are `def`, **not** `async def`. LightGBM inference and the Open-Meteo call are
blocking and CPU/IO-bound; FastAPI runs sync handlers in its threadpool, so a slow live
forecast doesn't stall the event loop. Writing `async def` around blocking code is the
single most common way this class of API falls over, and at 1–5 users the threadpool is
the entire concurrency story.

---

## 2. Endpoints

| Method | Path | Backing call | Notes |
|---|---|---|---|
| GET | `/health` | none | Liveness. No disk, no models. Always 200 if the process is up. |
| GET | `/ready` | mount check | 200 only if `models/`, `data/`, boundaries cache all resolve. This is what a container healthcheck hits. |
| GET | `/districts` | `district_registry()` + `load_district_fleets()` | Name, kind (`landkreis`/`stadt`), wind/solar nameplate MW. `?geometry=true` adds a simplified boundary polygon per district (§3a). |
| GET | `/forecast` | aggregate of all 18 | State total. Same envelope, `district: "Brandenburg (Total Aggregate)"`, plus a `districts[]` array of per-district energy totals with intervals — the un-summed inputs to the total (§3a). Feeds the choropleth in one call. |
| GET | `/forecast/{district}` | `forecast_district()` | The main endpoint. Envelope in §3. |
| GET | `/skill` | `results/macro_cv_metrics.csv`, `results/municipal_validation_metrics.csv` | Macro nMAE + PICP per (model, tech), and realized municipal PICP. |

`/forecast` and `/forecast/{district}` accept `?source=auto|batch|live` (default `auto`)
and `?mode=identity|pooled` (default `identity`, passed straight to
`resolve_calibration`). No date parameter — the window is always the current local
calendar day via `local_day_window()`, matching `dashboard.py:190`. Historical replay is
a different product; see §8.

`/districts?geometry=true` has no other params — the boundaries file is effectively
immutable, so the response is ETagged off its mtime rather than cached per-request.

---

## 3. The forecast envelope

This is the contract, and it is where three of your robustness requirements live. Every
forecast response, batch or live, aggregate or single district:

```json
{
  "district": "Barnim",
  "issued_at": "2026-07-24T12:00:00Z",
  "window": { "start": "2026-07-23T22:00:00Z", "end": "2026-07-24T22:00:00Z", "tz": "Europe/Berlin" },
  "source": "batch",
  "generated_at": "2026-07-24T03:14:00Z",
  "stale": false,
  "nameplate_mw": { "wind": 412.7, "solar": 388.1 },
  "calibration": {
    "mode": "identity",
    "wind":  { "scale": 1.0, "offset": 0.0 },
    "solar": { "scale": 1.0, "offset": 0.0 },
    "note": "Uncalibrated district: identity downscaling (physics-trust). Not validated against DSO telemetry."
  },
  "model": {
    "name": "lightgbm_quantile",
    "quantiles": [0.1, 0.5, 0.9],
    "sha256": { "wind_q50": "a1b2c3d4", "solar_q50": "e5f6a7b8", "...": "..." }
  },
  "interval": { "nominal_coverage": 0.8, "realized_picp_municipal": { "wind": 0.61, "solar": 0.55 } },
  "hours": [
    { "time": "2026-07-23T22:00:00Z",
      "wind_lower_mw": 41.2, "wind_median_mw": 88.7, "wind_upper_mw": 143.9,
      "solar_lower_mw": 0.0, "solar_median_mw": 0.0, "solar_upper_mw": 0.0 }
  ]
}
```

Design notes, each tied to a requirement you selected:

- **Never a bare number.** `hours` entries carry `lower`/`median`/`upper` as one object.
  There is no endpoint that returns medians alone, and no `?quantile=` parameter that
  would let a consumer strip the band. The `interval` block restates that realized
  municipal PICP runs *below* the nominal 80% — the same honesty caption
  `dashboard.py:148` already shows — so an API consumer can't lose it by not reading the
  UI.
- **Calibration is never implicit.** `calibration.mode` plus the `note` makes the
  identity-fallback assumption a field in the payload, not a footnote. A governance user
  diffing two districts can see that neither is DSO-validated.
- **Auditable.** `model.sha256` (short digests of the actual `.pkl` files, hashed once at
  startup and cached), `generated_at`, and explicit `window` bounds are enough to
  reconstruct any published number later. Plus request logging, §6.
- Typed as a Pydantic response model, because this envelope is the contract consumers
  build against and it makes `/docs` an accurate spec for free. The *other* five
  endpoints return plain dicts — response models there would be boilerplate for data
  nobody integrates against.

### 3a. Choropleth support: `districts[]` on `/forecast`, `geometry` on `/districts`

The frontend renders the choropleth client-side (Plotly.js `choroplethmap`, no
matplotlib, no new endpoint family) from two calls:

**`GET /districts?geometry=true`** — adds a simplified boundary polygon per district.
Boundaries are static (`data/external/brandenburg_districts.geojson`, raw Nominatim
polygons, currently 2.3 MB unsimplified) and cached; the added cost is one line —
`gdf.assign(geometry=gdf.geometry.simplify(0.002)).to_json()` in `src/data/districts.py`
— which brings the whole payload under ~150 KB (a fixed statewide-zoom map has no use for
sub-pixel detail). `preserve_topology=True` (the default) keeps every polygon valid;
independently-simplified neighbours can drift by up to `0.002°` (~220 m) at a shared
border, invisible at this map's zoom. Response is ETagged off the boundaries file's mtime
— it changes only when `fetch_district_boundaries()` re-runs, which is effectively never.

**`GET /forecast`'s `districts[]` array** — the per-district energy totals the state
total is already summed from, returned instead of discarded:

```json
"districts": [
  { "name": "Barnim",
    "nameplate_mw":     { "wind": 412.7, "solar": 388.1 },
    "wind_energy_mwh":  { "lower": 610.4, "median": 1180.2, "upper": 1902.8 },
    "solar_energy_mwh": { "lower": 210.1, "median":  455.7, "upper":  702.3 },
    "total_energy_mwh": { "lower": 820.5, "median": 1635.9, "upper": 2605.1 } }
]
```

Present only on `/forecast` (absent on `/forecast/{district}`, which is already a single
district). Both the batch aggregation (`dashboard.py:201-202`, two groupbys on one frame)
and the live aggregation (`dashboard.py:204-214`, sum of per-district frames) already
compute this breakdown before summing it away — returning it costs nothing extra to
compute, and it's the same frame and the same `issued_at` as the total, so the two
physically cannot disagree. The array is energy totals only, not hourly series (a
district's hourly chart is `/forecast/{district}`) — shipping 18 hourly series would send
the whole underlying parquet for a view that renders one scalar per district. Each figure
carries its interval, never a bare number, same as §3's rule for `hours[]`; note that
summing q10 across hours is the same comonotonic conservative-upper-bound assumption
`dashboard.py:170-175` already captions for the state band, now applied spatially too —
reuse that caption rather than inventing new wording.

The frontend draws the map from exactly these two requests, joining on
`properties.name` / `district.name` — proven to match on all 18 by
`dashboard.py:122`'s existing `featureidkey='properties.name'` trace.

A static server-rendered PNG (the deleted `export_choropleth.py`, matplotlib) was
considered and rejected: it can't carry per-district intervals or the calibration note
(violates the "never a bare number" rule), Streamlit already proves the client-side
GeoJSON approach works today (`dashboard.py:115-128`), and reviving it would mean writing
a new module and adding matplotlib back to a container §7 deliberately strips it from.

---

## 4. Freshness rule (requirement #1)

New helper in `src/dashboard/forecast_service.py`, used by `api.py` **and**
`dashboard.py`:

```
batch_is_current(parquet_path, window) -> bool
    True iff the parquet exists and its time_utc index covers [window.start, window.end).
```

Resolution for `source=auto`:

1. Batch parquet covers the requested window → serve it, `source: "batch"`,
   `generated_at` = file mtime, `stale: false`.
2. Batch missing or does not cover the window → run live NWP, `source: "live"`,
   `generated_at` = weather fetch time.
3. Live fetch fails (Open-Meteo down, models unloadable) → **`503`**, with the stale
   batch's window in the error body so the caller knows what exists and why it wasn't
   used.

Step 3 is the consequence of you *not* selecting "always answers, even degraded": a
governance dashboard showing last week's wind as today's is worse than one showing an
error. `source=batch` explicitly is the only way to get out-of-window data, and it
returns `stale: true` with the age in the payload — you can ask for stale, you cannot be
handed it.

The same helper fixes `dashboard.py:187` in the same commit.

---

## 5. Auth

Now: **none** — bind `--host 127.0.0.1`. The network is the boundary.

Prepared for later, ~6 lines, wired on the router from day one:

```
API_KEY = os.environ.get("API_KEY")           # unset -> open, current behaviour

def require_key(x_api_key: str = Header(None)):
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(401)
```

Setting `API_KEY` in the environment turns the whole API private with no code change and
no redeploy of anything but the env. `compare_digest` rather than `==` because it's the
same length and constant-time; there's no reason to write the flimsy version.

CORS: `CORSMiddleware` gated on `ALLOWED_ORIGINS` (comma-separated, empty = disabled).
Three lines, and without it the first browser frontend you point at this will fail in a
way that takes an hour to diagnose.

When you go web-accessible, the additions are: TLS at a reverse proxy (not in the app),
`API_KEY` set, and a rate limit *if* the caller set is untrusted — the live-NWP path
costs Open-Meteo quota per request, which is the actual abuse surface here, not CPU.

---

## 6. Caching, concurrency, logging

- `load_district_fleets()` reads both MaStR CSVs and does an sjoin — expensive, and
  currently `@st.cache_resource` in the dashboard. In the API: `@lru_cache` on a zero-arg
  loader, warmed during `lifespan` startup so the first request isn't the slow one.
- Forecast results: `@lru_cache(maxsize=64)` keyed by `(district, window_start, mode)`.
  The key rotates naturally at local midnight, stale entries fall out of the LRU. This is
  the replacement for `st.cache_data(ttl=1800)`. No Redis, no TTL bookkeeping.
- Model files: loaded per call by `_downscale_interval` via `joblib.load`
  (`forecast_service.py:94`) — six `.pkl` loads per forecast. At 1–5 users that's fine
  and I'd leave it. If `/forecast` latency ever bothers you, `@lru_cache` on a
  `load_model(tech, q)` helper is the one-line fix, and it belongs in `forecast_service`
  so the dashboard gets it too.
- Request logging: `loguru` is already a dependency. Log method, path, district,
  `source`, `generated_at`, status, duration — one middleware, JSON lines to
  `logs/api.log`. That plus the payload's `model.sha256` is your audit trail.

---

## 7. Deployment

**Standalone (start here):**

```
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

One worker. At 1–5 users, multiple workers only multiply the memory held by cached
fleets and models for no throughput you'd notice.

**Docker (worth doing now — it's ~15 lines):**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group research
COPY src/ src/
COPY api.py conf/ ./
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run with the three gitignored directories mounted read-only, plus a writable `logs/`
mount for the §6 request-log audit trail:

```
docker run -p 8000:8000 \
  -v $PWD/models:/app/models:ro \
  -v $PWD/data:/app/data:ro \
  -v $PWD/results:/app/results:ro \
  -v $PWD/logs:/app/logs \
  energy-api
```

All three of `models/`, `data/`, `results/` are read-only for the `api` service — no
request handler writes to disk (the `/choropleth.png` writer this mount used to justify
is dropped; see §3a). `logs/` is the one writable mount. The `batch` service (§13) still
mounts `results/` read-write, since it writes the archive parquet.

Because `data/` is now strictly read-only in the container, the boundaries cache must be
warm before deploy — `/ready`'s boundary check should be a plain `os.path.exists()` so a
cold cache fails loudly at startup rather than mid-request, 18 Nominatim calls deep.

**The `pyproject.toml` change this needs** (finding (c)): move `torch`,
`pytorch-forecasting`, `pytorch-lightning`, `torchmetrics`, `shap`, `interpret`,
`jupyter`, `seaborn` into a `[dependency-groups] research` group. Training and notebooks
run `uv sync` as today; the container runs `--no-group research` and drops from ~6 GB to
roughly 400 MB. No source changes — nothing in the API import path touches those
packages. `libgomp1` is in the apt line because LightGBM needs it on slim.

`/ready` failing on a missing mount is what makes the volume requirement self-diagnosing
instead of a 500 three clicks into the dashboard.

---

## 8. Deliberately not in scope

| Skipped | Add when |
|---|---|
| Historical / arbitrary-date forecast replay | Someone needs to audit a *past* forecast against actuals. Needs archived issue-time forecasts, which the pipeline doesn't retain — a storage design, not an endpoint. |
| ~~`/choropleth.geojson`~~ — **resolved** | Superseded: the frontend renders its own map (Plotly.js, client-side), so this landed as `/districts?geometry=true` rather than a new endpoint — see §3a. Boundaries are simplified (`shapely.simplify(0.002)`, ~150 KB) rather than shipped raw (2.3 MB unsimplified). |
| Rate limiting / per-caller keys | The API leaves the trusted network. Quota on the live-NWP path is the thing to limit, not requests. |
| Database, ORM, migrations | Never, on current requirements. Parquet + CSV on disk *is* the store. |
| Async handlers, workers, task queue | Sustained concurrency past ~20, measured not guessed. |
| Pydantic models for the non-forecast endpoints | A second consumer integrates against them. |
| WebSocket / push updates | Forecasts change hourly at most; polling is correct. |

---

## 9. Implementation order

1. `batch_is_current()` in `forecast_service.py` + wire into `dashboard.py`. **Fixes the
   live stale-data bug independently of the API.** Ship this first, it stands alone.
2. `api.py`: `/health`, `/ready`, `/districts` (+ `?geometry=true`). Proves mounts and
   fleet loading.
3. `/forecast/{district}` with the full envelope + the auto/batch/live/503 resolution.
4. `/forecast` aggregate (+ `districts[]` breakdown, §3a), `/skill`.
5. `pyproject.toml` dependency-group split, then `Dockerfile`.

New dependency: `uv add "fastapi[standard]"` (brings uvicorn).

**Self-check** — one `test_api.py`, FastAPI's `TestClient`, no fixtures, four asserts:
`/health` is 200; `/forecast/Barnim` returns 24 hours with `lower ≤ median ≤ upper` and
all MW ≥ 0; a parquet whose window doesn't cover today never comes back as
`source: "batch", stale: false`; and `/districts` returns exactly 18. That last pair is
the whole robustness contract in two assertions.

---

## 10. Nightly 04:00 refresh

### 10.1 First, which job actually refreshes the dashboard

`run_pipeline.sh` already runs nightly and calls
`src.validation.extend_kw_weather --all`. That is **not** the job you want here. It
extends the *archive* (ERA5 reanalysis) weather parquets used for training and municipal
validation. The dashboard's numbers come from Open-Meteo's *forecast* endpoint via
`fetch_forecast_weather()` (`forecast_service.py:45`), which is called fresh at inference
and caches nothing to disk. Extending the archive changes nothing the dashboard displays.

What the dashboard serves is `results/forecasts/brandenburg_state_forecast.parquet`.
So the 04:00 job is: **regenerate that parquet**, which internally does the Open-Meteo
forecast fetch for all 18 districts. Two independent nightly jobs, different purposes —
keep both, don't merge them.

### 10.2 Blocker: the batch generator is a notebook

Today the parquet is produced by `jupyter nbconvert --execute --inplace notebooks/11_...`.
That can't be the cron mechanism in the container, for three reasons:

1. §7 drops `jupyter` from the image. Putting it back to run a cron job costs the whole
   size saving.
2. `--inplace` mutates a version-controlled notebook on every run, so the nightly job
   produces a git diff every day.
3. Failure semantics are poor — cell 5 catches per-district exceptions and `continue`s.

**Fix, and it's small:** move cells 3/5/7 into `src/dashboard/batch_forecast.py` with a
`run()` entry point. This is ~25 lines lifted almost verbatim, and it's what CLAUDE.md
already mandates: *"Notebook 11 orchestrates batch inference by calling `src/` utilities
— pipeline logic stays in `src/`, not the notebook."* The notebook keeps its narrative
and plots and calls `run()` for the heavy part; cron calls the same function. One code
path, two callers.

```
uv run -m src.dashboard.batch_forecast     # what cron runs
```

### 10.3 Two correctness holes to close while moving it

**Partial batches are silently published.** Cell 5 skips failed districts; cell 7 asserts
only that each *present* district has a full horizon and that there are no NaNs. It never
asserts there are 18. If six districts fail their Open-Meteo call, a 12-district parquet
is written and the dashboard reports a state total ~⅓ too low with no indication anything
went wrong — the aggregate is a *sum*, so a missing district looks exactly like a quiet
day. Add `assert state.district.nunique() == 18` before the write, and exit non-zero.
Given your "never a silently wrong number" requirement, a failed refresh that leaves
yesterday's parquet in place is strictly better: §4's freshness guard then sees it as
out-of-window and the API falls through to live NWP or 503.

**Non-atomic write.** `to_parquet(OUT_PATH)` writes in place. A crash or an OOM mid-write
leaves a truncated file that the API will happily try to read. Write to
`OUT_PATH + ".tmp"` then `os.replace()` — atomic on POSIX, one extra line, and it means a
reader never observes a half-written forecast.

### 10.4 Scheduling

**Host cron → one-shot container.** No scheduler inside the API container.

```cron
0 4 * * *  cd /home/ced/bachelor/final && docker compose run --rm batch >> logs/cron_batch.log 2>&1
```

```yaml
# compose.yaml
services:
  api:
    build: .
    ports: ["127.0.0.1:8000:8000"]
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
      - ./results:/app/results:ro   # was rw, solely for the now-dropped /choropleth.png
      - ./logs:/app/logs            # rw — §6 request log is the audit trail
    restart: unless-stopped

  batch:                      # same image, different command, run to completion by cron
    build: .
    command: ["uv", "run", "-m", "src.dashboard.batch_forecast"]
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
      - ./results:/app/results
    profiles: ["cron"]        # never started by `docker compose up`
```

The API container is never restarted or signalled. It re-reads the parquet through the
shared `results/` volume; §6's forecast cache is keyed by `(district, window_start, mode)`
and the window rolls at local midnight, so the 04:00 output is picked up on the next
request without any invalidation logic. The `/ready` mount check already covers the
volume being present.

Why host cron rather than supercronic/APScheduler in the image: one job, once a day, on a
box that already runs cron for `run_pipeline.sh`. An in-container scheduler adds a
supervised second process and a new failure mode (job runs, container gets OOM-killed,
nobody notices) to replace something the OS does correctly. If this ever moves to k8s,
`batch` becomes a `CronJob` with the same command — that's the migration, and it's why
the batch job is a separate service rather than a thread in `api.py`.

### 10.5 Timing notes

- **04:00 local produces a window that is 4 h in the past.** `local_day_window()` returns
  `[00:00, 24:00)` of the *current* local day, so a 04:00 run publishes a parquet whose
  first four hours have already happened. That is what the dashboard does today and is
  probably what you want (a full calendar-day view for a governance audience), but if you
  want 24 h strictly *ahead*, the run has to move to ~23:30 the night before and target
  tomorrow's window. Worth deciding explicitly — it's a one-line change to the window
  argument, and hard to notice is wrong.
- **DST is handled**, because `local_day_window()` hardcodes `Europe/Berlin` rather than
  reading the container clock. The container needs no `TZ` setting; host cron fires at
  04:00 host-local, which is the only DST-sensitive part.
- **Quota is a non-issue**: one run/day × 18 districts, batched per district with
  `pause=1.0`. Roughly 18 calls, against the quota accounting added in `98dd5f2`.
- **Existing crontab** (both jobs *are* installed; add the 04:00 batch line as a third):

  ```cron
  */15 * * * *  cd /home/ced/bachelor/final && .venv/bin/python run_scraper.py >> data/scraper.log 2>&1
  0 * * * *     /home/ced/bachelor/final/run_pipeline.sh
  ```

  Note `run_pipeline.sh` runs **hourly**, not nightly as CLAUDE.md states. Harmless —
  `extend_kw_weather --all` extends the archive through *yesterday*, so 23 of the 24
  daily runs are no-ops after the first success — but the doc is wrong and the schedule
  is doing 24× the work it needs to. Worth either fixing the docstring or moving it to
  `0 3 * * *`.

### 10.6 Monitoring, minimally

The parquet's own mtime is the health signal — `/ready` can report it and the dashboard
already surfaces staleness through §4. If a silent failure for several days would matter,
the smallest real check is one line appended to the cron entry: `|| echo "batch forecast
FAILED $(date)" >> logs/cron_batch.log`, and have `/ready` return degraded when the
parquet is older than 25 h. Skip alerting/email/Prometheus until a missed refresh has
actually cost you something.

---

## 11. The two products

Target: (1) a 24–48 h forecast, (2) a live monitor in the spirit of
`energiemonitor.e-dis.de/koenigs-wusterhausen`.

The good news is that **neither needs new model code**. Both are
`forecast_from_fleet()` (`forecast_service.py:104`) with a different `window`. The
existing call already fetches `forecast_days=2, past_days=1` — 72 h of weather — and then
throws most of it away by slicing to today. Past hours, current hour, and +48 h are all
sitting in the response you already pay for.

### 11.1 Product 2 is two different things wearing one UI

| Region | What "live" can mean | Source |
|---|---|---|
| Königs Wusterhausen, Nauen | **Measured**, 15-min | `data/webscrape_{kwusterhausen,nauen}.csv` — already scraped every 15 min, current as of today 12:00, 5216 rows back to 2026-05-28 |
| The other 16 districts | **Modelled nowcast**, hourly | `forecast_from_fleet(window=(now-24h, now+1h))` |

For the two pilots you are not estimating anything — you are re-serving the same
measurement E.DIS shows, and the scraper CSV even carries their full payload
(`photovoltaik`, `windkraft`, `biomasse`, plus consumption split and
`heute_eingespart_tCO2`). That is a genuine live monitor and it's a CSV read.

For everywhere else there is no telemetry, so "live" is a model output. **These must not
share a series without labels.** This is the same principle as §3's calibration field: a
modelled hourly nowcast rendered in live-meter visual language claims a measurement
precision that does not exist. Concretely, from your own `results/nwp_gap_report.md`,
forecast weather costs **+90.9% wind MAE and +76.8% solar MAE** against reanalysis — the
nowcast is roughly twice as wrong as the numbers your validation plots show, and per
`results/municipal_validation_metrics.csv` the 80% band already under-covers at Gemeinde
scale. A dial reading "1,569 kW" implies neither.

So the contract types them:

```
GET /live/{region}
{
  "region": "Königs Wusterhausen",
  "observed": {                 // null for the 16 districts with no telemetry
    "kind": "measured", "resolution_min": 15, "source": "E.DIS Energiemonitor (scraped)",
    "as_of": "2026-07-24T12:00:00Z",
    "series": [{ "time": "...", "wind_kw": 1569.06, "solar_kw": 3166.45, "biomass_kw": 0.0 }]
  },
  "modelled": {
    "kind": "modelled", "resolution_min": 60, "as_of": "2026-07-24T12:00:00Z",
    "caveat": "Model estimate, not measurement. Forecast-weather inputs carry +91% wind / +77% solar MAE vs reanalysis (results/nwp_gap_report.md); 80% interval under-covers at municipal scale.",
    "series": [{ "time": "...", "wind_median_mw": ..., "wind_lower_mw": ..., "...": ... }]
  }
}
```

One endpoint, `observed: null` where there's no telemetry. A consumer physically cannot
plot a measured value without knowing it's measured.

**The pilot bonus, nearly free:** for KW and Nauen both blocks are populated, so the live
view *is* a continuously-updating validation of the downscaling contract — measured vs
modelled on one axis, refreshing every 15 minutes. That's the strongest thing in this
whole system for a governance audience and for the thesis, and it costs one extra series
in a chart because both halves already exist.

### 11.2 Cadence

- Telemetry: 15-min, already on cron. `/live` reads the CSV tail per request — 5216 rows
  is nothing, no cache needed beyond an mtime check.
- Model nowcast: **hourly**, `@lru_cache` keyed by `(region, hour)`. The features are
  hourly and Open-Meteo refreshes hourly, so recomputing every 15 min would produce a
  chart that moves without new information. Let the measured series carry the 15-min
  motion; that's the half that actually has it.
- No websockets. Poll `/live` every 60 s client-side; the payload is a few KB.

### 11.3 On extending to 48 h — **superseded by §12**, kept for the reasoning

Code-wise this is a parameter — `fetch_forecast_weather` already returns 48 h and
`local_day_window()` is what limits you to today. Replace the window with
`(now.ceil('h'), now + 48h)` and it works.

The caveat is real though, and it's the one place I'd not just ship it: `nwp_gap`
reports a **single pooled number across all lead times**. Skill decays with lead time, so
hour 47 is materially worse than hour 1, and the q10/q90 models were fit at macro scale
on reanalysis-quality features — the band is already too narrow municipally and does not
widen with horizon at all. A 48 h chart with a constant-width interval is exactly the
"number without an honest interval" you ruled out.

Cheapest fix that makes it defensible: add a lead-hour `groupby` to
`src/evaluation/nwp_gap.py` (same module, same data, one extra dimension) to get
error-vs-lead-time. Then either widen the published interval by the measured
lead-time factor, or ship 48 h with a per-lead-time caveat in the payload and a visual
break at +24 h. Either is honest; a flat band to +48 h is not.

### 11.4 What this does to the plan

Ordering stays as §9, plus:

6. `/live/{region}` with `observed`/`modelled` blocks — telemetry half first (it's a CSV
   read and immediately useful), nowcast half second.
7. Lead-time stratification in `nwp_gap`, then extend the forecast window to 48 h.

No new dependency, no new model, no second service. Products 1 and 2 are the same
inference function called with three different windows (past / now / ahead), which is why
this stays two files rather than becoming a platform.

---

## 12. Rolling 24 h + today's overview (supersedes §11.3)

Agreed shape: a **rolling next-24 h** forecast that always starts from now, plus a
**today's overview** on the fixed local calendar day. Product 2 deferred.

This is a better design than the 48 h extension and I'd take it on the merits, not just
because it's less work: every displayed hour stays inside the 0–24 h lead range your
existing validation actually covers, so nothing needs new evidence to be defensible. The
§11.3 problem — a flat interval stretched to +48 h where skill has decayed and the band
doesn't widen — simply stops existing.

Both views already exist in code. `forecast_from_fleet` (`forecast_service.py:120-124`)
has the rolling branch written and currently unused, because `dashboard.py:190` always
passes `window=local_day_window()`:

```python
now = pd.Timestamp.now(tz='UTC').ceil('h')
X = X[X.index >= now].head(horizon_h)     # rolling — the dead branch
```

### 12.1 The constraint: hourly refresh does not fit the free tier

I counted the weather nodes. **535 across the 18 districts** (Uckermark 52, Potsdam-Mittelmark 48,
down to 4 for Frankfurt (Oder)).

Open-Meteo bills per *location*, not per HTTP request (`weather_ingestion.py:127`), at
1.0 per location for ≤10 variables and ≤2 weeks — the forecast path uses 5 variables over
3 days, so it's 1.0 flat. Therefore:

| Refresh | Calls/day | vs 10,000 free limit |
|---|---|---|
| Hourly | **12,840** | ✗ 128% — over |
| Every 2 h | 6,420 | 64% |
| Every 3 h | 4,280 | 43% |
| 4×/day (NWP-aligned) | 2,140 | 21% |
| 1×/day (§10 as written) | 535 | 5% |

Batching all 535 nodes into one request saves nothing — cost scales with locations
regardless of how you group them.

### 12.2 The fix: decouple the slice rate from the fetch rate

You don't need to *fetch* hourly to *show* a rolling window hourly. Two reasons this is
free:

1. `fetch_forecast_weather` already requests `past_days=1` + `forecast_days=2`, so one
   response spans ~72 h. A rolling 24 h window is a **slice** of data you already hold.
2. NWP models don't update hourly anyway. ICON and ECMWF run 4×/day (00/06/12/18 UTC).
   Refetching at 13:00 when the 12Z run landed an hour ago returns *the same numbers* —
   you'd spend 535 calls to redraw an identical chart.

So:

- **Fetch + infer every 3 h** (or NWP-aligned at 01/07/13/19 UTC, ~40 min after each run
  publishes). Writes one parquet spanning `[today 00:00 local, issue + 48 h]`.
- **Both views are request-time slices of that one parquet:**
  - *Next 24 h* = rows in `[now.ceil('h'), now + 24 h)` — **rolls every hour at zero API
    cost**, which is exactly the behaviour you asked for
  - *Today's overview* = rows inside the local calendar day, i.e. today's
    `local_day_window()`, unchanged from what the dashboard shows now

One stored frame, two slices, ~4,280 calls/day at 3-hourly. The rolling view updates
hourly regardless, because the slice boundary moves even when the data doesn't.

**One required change:** bump `forecast_days` from 2 to **3** in `fetch_forecast_weather`.
At 22:00 local the rolling window reaches into the hour after tomorrow's end, which
`forecast_days=2` doesn't cover. Three days is still ≤2 weeks, so **the cost stays 1.0 per
location** — the fix is free (`test_weather_ingestion.py:13` confirms 7 days at 5 vars
still costs 1.0).

If you genuinely want true hourly fetches, the answer is an Open-Meteo API key (paid tier)
— not a code change. I'd not bother: you'd be paying to re-download identical NWP output
three hours out of four.

### 12.3 API surface

`?view=rolling|today` on the existing endpoints, rather than new paths — same envelope,
same district list, only the window differs.

```
GET /forecast/{district}?view=rolling   # default: [now, now+24h)
GET /forecast/{district}?view=today     # local calendar day
```

The §3 envelope already carries `window.start`/`window.end` and `generated_at`, so a
consumer can always tell which view it got and how old the underlying run is. Add
`view` and `lead_hours` (`[1, 24]` or `[-8, 15]` for a mid-day "today" slice) so the
lead-time range is explicit — that's the field that keeps the interval honest without
needing the §11.3 stratification work.

`@lru_cache` key becomes `(district, issue_time, mode)` where `issue_time` is the
parquet's generation timestamp — not the request hour. Two requests in the same 3-hour
window hit the same cached inference; the *slice* is cheap and recomputed per request.

### 12.4 Revisions to §10

- Cron becomes `20 */3 * * *` (staggered off `run_pipeline.sh` at `0 * * * *`), not
  `0 4 * * *`.
- §4's freshness rule tightens: batch is current if `now - generated_at < 4 h` **and** the
  parquet covers the requested slice. A dead refresh job is now caught within one cycle
  instead of a day.
- The §10.3 fixes (assert 18 districts, atomic `os.replace` write) matter *more* at 8
  runs/day than at 1 — eight chances a day to publish a partial state total.
- Retaining each run with an `issued_at` column is **in** — see §13.2.

---

## 13. Decisions locked: NWP-aligned refresh + issued_at archive

### 13.1 Schedule

Four runs/day aligned to the 00/06/12/18 UTC model cycles, offset **+4 h** to sit safely
after publication:

```cron
CRON_TZ=UTC
20 4,10,16,22 * * *  cd /home/ced/bachelor/final && docker compose run --rm batch >> logs/cron_batch.log 2>&1
```

Cost: **2,140 calls/day, 21% of the free 10,000** — leaving room for `/live` nowcasts when
Product 2 lands.

Two details behind the numbers:

- **Why +4 h and not +40 min.** I don't know Open-Meteo's exact per-model publication lag,
  and a too-eager fetch silently re-downloads the *previous* run — the worst outcome here,
  because it looks like a successful refresh. Four hours is comfortably past ICON-EU and
  IFS publication. If you want to tighten it, the empirical check is cheap: fetch one node
  hourly for a day and diff consecutive responses to see when values actually change. Not
  worth doing unless the 4 h offset visibly bothers you.
- **`CRON_TZ=UTC` matters.** Host cron runs Europe/Berlin, so local-time slots would drift
  an hour against the UTC model cycles across DST. Ubuntu's cron supports `CRON_TZ`; if it
  ever doesn't, fall back to local-time slots and accept the drift — the 4 h margin
  absorbs it.

Convenient side effect: the 04:20 UTC run lands ~06:20 local, so a fresh forecast is
waiting at the start of the working day.

### 13.2 Archive

**One file per run, never appended:**

```
results/forecasts/archive/2026-07-24T1600Z.parquet
```

Write-once files rather than a monthly append — no read-modify-write, no corruption
window, no growing rewrite cost, and `pd.read_parquet("results/forecasts/archive/")` reads
the whole directory as one dataset via pyarrow. Partitioned-dataset semantics for free.

Schema: the existing forecast columns plus an `issued_at` (UTC, constant within a file).
`lead_hours` is *not* stored — it's `time_utc - issued_at`, derive it at analysis time.

Volume: 18 districts × ~48 h ≈ 864 rows/run ≈ 60 KB, × 4/day ≈ **~90 MB/year**. No
retention policy needed; revisit if it ever matters.

**The payoff** — this is what makes the deferred §11.3 work possible on your own system:

```
archive ⋈ 50Hertz actuals (macro) or KW/Nauen telemetry (municipal)  on time_utc
  → groupby(lead_hours) → the skill-vs-lead-time curve, measured not inferred
```

Also gives you honest reliability diagrams per lead time, and it's the only way to ever
answer "was the forecast we published last Tuesday any good?" — which is the auditability
requirement from §3 in its strongest form.

### 13.3 One consequence worth taking: drop the separate "current" file

With an archive, `results/forecasts/brandenburg_state_forecast.parquet` becomes a second
copy of data that already exists, and two writes per run that can disagree. Replace it
with a resolver in `forecast_service.py`:

```
latest_forecast_path() -> newest results/forecasts/archive/*.parquet  (or None)
```

Then the batch job writes **one** file, and `dashboard.py:29` — which currently hardcodes
the path — calls the resolver instead. Same root-cause shape as §4's freshness guard: one
place, all callers.

Filenames sort lexicographically in issue order (`...T0400Z` < `...T1000Z`), so "newest"
is `max(glob(...))` — no mtime dependence, which matters because Docker volume copies
don't preserve mtimes reliably. Chose glob-newest over a `current.parquet` symlink
deliberately: symlinks across bind mounts are a class of problem this doesn't need.

### 13.4 Revised implementation order

Replaces §9's tail:

1. `batch_is_current()` + wire into `dashboard.py` — **fixes the live stale-data bug, ships alone**
2. `src/dashboard/batch_forecast.py` extracted from notebook 11, with: assert-18-districts,
   atomic `os.replace`, `issued_at` column, archive-file naming
3. `latest_forecast_path()` resolver; repoint `dashboard.py`
4. `forecast_days` 2 → 3 in `fetch_forecast_weather` (free — cost stays 1.0/location)
5. `api.py`: `/health`, `/ready`, `/districts` (+ `?geometry=true`, §3a)
6. `/forecast/{district}?view=rolling|today` with the §3 envelope + `view`/`lead_hours`
7. `/forecast` aggregate (+ `districts[]` breakdown, §3a), `/skill`
8. `pyproject.toml` dependency-group split → `Dockerfile` → `compose.yaml`
9. Install the `CRON_TZ=UTC` cron line

Steps 1–4 are independent of FastAPI entirely and improve the current Streamlit dashboard
on their own. If the API stalls, that work is not wasted.

**Added self-check** for step 2, alongside §9's: given two archive files, the resolver
returns the lexicographically newest; and a run that produces fewer than 18 districts
raises before writing anything.

**Added self-check** for steps 5/7: `/districts?geometry=true` returns 18 valid,
non-self-intersecting polygons totaling under 300 KB; `/forecast`'s `districts[]` sums to
the state total's own energy figure within float tolerance — the single-source-of-truth
property as a test.
