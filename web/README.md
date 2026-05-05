# PyTGS Web

A browser-based UI for fitting Transient Grating Spectroscopy signals with PyTGS. Runs entirely on the user's machine via [Pyodide](https://pyodide.org/) — no backend, no uploads.

## What it does

- Calibration: derive grating spacing from a reference POS/NEG pair given a known SAW velocity.
- Batch queue: drop multiple POS/NEG pairs, run fits in parallel across Web Workers, view results + plots + CSV export.
- Config editor: full control over `signal_process`, `fft`, `lorentzian`, `tgs`, and `plot` options from `config.yaml`.
- PSFC-inspired styling to match [psfc.mit.edu](https://www.psfc.mit.edu/).

## How it works

```
Browser (Next.js SPA, static)
  └── N Web Workers (one per CPU core)
        └── Pyodide runtime
              ├── numpy, scipy, pandas, pyyaml
              └── PyTGS pytgs/ (mounted from /public/pytgs/)
```

The first page visit loads ~30 MB of Pyodide runtime + scientific packages; subsequent visits are instant from browser cache. Individual fits run 2–3× slower than native Python, but batches are parallelized across workers so a 100-file batch on an 8-core laptop completes in ~5–10 min (vs native ~10–15 min).

## Develop

```bash
cd web
npm install
npm run dev
```

The `predev` / `prebuild` npm hooks automatically copy the PyTGS Python sources from `../pytgs/` into `public/pytgs/` so Pyodide can fetch them at runtime.

Open http://localhost:3000.

## Build (static export)

```bash
cd web
npm run build
```

Output lands in `web/out/` — plain HTML/JS/CSS/PNGs, deployable to any static host.

## Deploy to Vercel

Vercel auto-detects the setup from the repo-root `vercel.json`. Push to a GitHub repo connected to a Vercel project; static output is deployed to the project domain.

```bash
# One-off deploy from CLI
vercel deploy
```

## Deploy to GitHub Pages / Cloudflare Pages / Netlify / S3

Any static host works — just serve the `web/out/` directory. The app has no server-side dependencies.

## Verification

Run the example fit through the UI to compare against the canonical Python output:

1. Load `examples/Cu-110-2024-05-01-03.40um-cooldown-POS-1.txt` + `NEG-1.txt` (and friends) into the batch queue.
2. Click **Run batch**.
3. Compare the extracted parameters against `examples/fit/fit.csv` — they should match to numerical-noise tolerance.

## Architecture notes

- **State management**: React Context (`lib/app-store.tsx`). No external state library — the app has a bounded number of stateful components.
- **Worker pool**: `lib/pyodide-pool.ts`. Default size = `navigator.hardwareConcurrency - 1`, capped at 8. Lazy-spawn on first use; prewarms after init.
- **File pairing**: `lib/file-pairing.ts` parses `{prefix}-{study}-{POS|NEG}-{index}.txt` to auto-match drops.
- **Plotting**: drawn client-side with Plotly from raw arrays returned by the worker — the Python kernel no longer touches matplotlib.
- **No modifications to `pytgs/`**: the worker mounts the package read-only and only sets env vars + warning filters before importing.

## Known constraints

- Pyodide has no pthreads by default, so each fit is single-threaded. Parallelism comes from running many workers, not from threading within one worker.
- Browsers cache Pyodide aggressively after first visit, but cold starts on new machines are ~30 MB of download.
- `scipy.optimize.curve_fit` with `maxfev=1_000_000` works but is noticeably slower than native. Consider reducing `maxfev` in the config editor if you see slow fits.
