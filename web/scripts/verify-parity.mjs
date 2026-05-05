#!/usr/bin/env node
// Runs the same code path the browser worker uses, but in Node via the
// Pyodide Node build. Compares extracted parameters against the canonical
// examples/fit/fit.csv values.

import { loadPyodide } from "pyodide";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(__dirname, "..");
const repoRoot = resolve(webRoot, "..");
const pytgsDir = join(webRoot, "public", "pytgs");

const manifest = JSON.parse(
  await readFile(join(pytgsDir, "manifest.json"), "utf8"),
);

const config = {
  path: "examples",
  study_names: ["cooldown"],
  idxs: null,
  signal_process: {
    heterodyne: "di-homodyne",
    null_point: 2,
    initial_samples: 50,
    baseline_correction: { enabled: false, pos: null, neg: null },
  },
  fft: { signal_proportion: 1.0, analysis_type: "fft" },
  lorentzian: {
    signal_proportion: 1,
    frequency_bounds: [0.1, 0.9],
    dc_filter_range: [0, 12000],
    bimodal_fit: false,
    use_skewed_super_lorentzian: false,
  },
  tgs: { grating_spacing: 3.5276, signal_proportion: 1, maxfev: 1000000 },
};

console.log("→ Loading Pyodide…");
const pyodide = await loadPyodide();

console.log("→ Installing packages…");
await pyodide.loadPackage(["numpy", "scipy", "pandas", "pyyaml"]);

console.log("→ Mounting PyTGS sources…");
for (const rel of manifest.files) {
  const content = await readFile(join(pytgsDir, rel), "utf8");
  const dir = rel.substring(0, rel.lastIndexOf("/"));
  if (dir) {
    try {
      pyodide.FS.mkdirTree(dir);
    } catch (_) {}
  }
  pyodide.FS.writeFile(rel, content);
}

pyodide.runPython(`
import os, sys, warnings
os.environ["PYTGS_HEADLESS"] = "1"
sys.path.insert(0, "/")
warnings.filterwarnings("ignore")
from pytgs.tgs import tgs_fit
`);

// Load the example files into Pyodide FS
const pos = await readFile(
  join(repoRoot, "examples", "Cu-110-2024-05-01-03.40um-cooldown-POS-1.txt"),
);
const neg = await readFile(
  join(repoRoot, "examples", "Cu-110-2024-05-01-03.40um-cooldown-NEG-1.txt"),
);
pyodide.FS.mkdirTree("/tmp/job");
pyodide.FS.writeFile("/tmp/job/Cu-110-2024-05-01-03.40um-cooldown-POS-1.txt", pos);
pyodide.FS.writeFile("/tmp/job/Cu-110-2024-05-01-03.40um-cooldown-NEG-1.txt", neg);

pyodide.globals.set("_web_config_json", JSON.stringify(config));

console.log("→ Running tgs_fit on Cu-110-POS-1…");
const t0 = Date.now();
pyodide.runPython(`
import json
from pytgs.tgs import tgs_fit

_pos = "/tmp/job/Cu-110-2024-05-01-03.40um-cooldown-POS-1.txt"
_neg = "/tmp/job/Cu-110-2024-05-01-03.40um-cooldown-NEG-1.txt"

_cfg = json.loads(_web_config_json)
_result = tgs_fit(
    _cfg, _pos, _neg,
    grating_spacing=_cfg["tgs"]["grating_spacing"],
    signal_proportion=_cfg["tgs"]["signal_proportion"],
    maxfev=_cfg["tgs"]["maxfev"],
)
(start_idx, start_time, grating_spacing,
 A, A_err, B, B_err, C, C_err,
 alpha, alpha_err, beta, beta_err,
 theta, theta_err, tau, tau_err,
 f, f_err, *_rest) = _result
_out_json = json.dumps({
    "start_idx": int(start_idx),
    "start_time": float(start_time),
    "A": float(A), "A_err": float(A_err),
    "B": float(B), "B_err": float(B_err),
    "C": float(C), "C_err": float(C_err),
    "alpha": float(alpha), "alpha_err": float(alpha_err),
    "beta": float(beta), "beta_err": float(beta_err),
    "theta": float(theta), "theta_err": float(theta_err),
    "tau": float(tau), "tau_err": float(tau_err),
    "f": float(f), "f_err": float(f_err),
})
`);
const result = JSON.parse(pyodide.globals.get("_out_json"));
const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

// Expected values from examples/fit/fit.csv row 1
const expected = {
  start_idx: 68,
  start_time: 3.4249999999999992e-9,
  A: 0.00126652902886191,
  B: 0.16926792606066376,
  C: -0.0025298107131092335,
  alpha: 0.00010439343310671782,
  beta: 0.025009616659827778,
  theta: -17.001895884822158,
  tau: 2.7936811280551525e-9,
  f: 542519649.8745747,
};

console.log(`\nFit complete in ${elapsed}s.\n`);
console.log("Param     | Expected                | Actual                  | Δ%");
console.log("----------|-------------------------|-------------------------|--------");

let allPass = true;
const tol = 0.05; // 5% tolerance — Pyodide uses OpenBLAS, native uses MKL; small differences expected
for (const k of ["start_idx", "start_time", "A", "B", "C", "alpha", "beta", "theta", "tau", "f"]) {
  const exp = expected[k];
  const act = result[k];
  const delta = exp !== 0 ? Math.abs((act - exp) / exp) * 100 : Math.abs(act - exp);
  const pass = delta < tol * 100;
  allPass = allPass && pass;
  const mark = pass ? "✓" : "✗";
  console.log(
    `${k.padEnd(10)}| ${exp.toExponential(6).padEnd(24)}| ${act.toExponential(6).padEnd(24)}| ${delta.toFixed(3)}% ${mark}`,
  );
}

console.log("\n" + (allPass ? "✓ All parameters within 5% tolerance." : "✗ One or more parameters outside tolerance."));
process.exit(allPass ? 0 : 1);
