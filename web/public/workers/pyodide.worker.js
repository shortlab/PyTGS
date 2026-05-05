// PyTGS Pyodide Web Worker.
// Runs numpy/scipy + PyTGS pytgs/ in the browser. Captures raw fit-curve data
// (no matplotlib rendering) so the UI can draw interactive plots with Plotly.
// Communication: postMessage({id, type, ...}) → {id, type, ...}

const PYODIDE_VERSION = "0.29.3";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

self.importScripts(PYODIDE_BASE + "pyodide.js");

let pyodide = null;
let initPromise = null;

function log(level, message) {
  self.postMessage({ type: "log", level, message });
}

function stringifyError(err) {
  if (err == null) return "Unknown error";
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.stack || err.message || err.toString();
  if (typeof err === "object") {
    if (typeof err.message === "string") return err.message;
    try {
      return JSON.stringify(err);
    } catch (_) {
      /* fall through */
    }
  }
  try {
    return String(err);
  } catch (_) {
    return "Unserializable error";
  }
}

async function initialize() {
  log("info", "Loading Pyodide runtime…");
  pyodide = await self.loadPyodide({
    indexURL: PYODIDE_BASE,
    stdout: (line) => log("info", line),
    stderr: (line) => log("warning", line),
  });

  log("info", "Loading scientific Python packages…");
  // matplotlib is not needed — we skip static plotting entirely.
  await pyodide.loadPackage(["numpy", "scipy", "pandas", "pyyaml"]);

  log("info", "Mounting PyTGS source files…");
  const manifestResp = await fetch("/pytgs/manifest.json");
  if (!manifestResp.ok) {
    throw new Error(`Failed to fetch /pytgs/manifest.json: ${manifestResp.status}`);
  }
  const manifest = await manifestResp.json();

  for (const rel of manifest.files) {
    const resp = await fetch(`/pytgs/${rel}`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch /pytgs/${rel}: ${resp.status}`);
    }
    const content = await resp.text();
    const dir = rel.substring(0, rel.lastIndexOf("/"));
    if (dir) {
      try {
        pyodide.FS.mkdirTree(dir);
      } catch (_) {}
    }
    pyodide.FS.writeFile(rel, content);
  }

  log("info", "Preparing environment…");
  pyodide.runPython(`
import os, sys, warnings, logging

os.environ["PYTGS_HEADLESS"] = "1"
if "/" not in sys.path:
    sys.path.insert(0, "/")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
`);

  log("info", "Importing PyTGS core modules…");
  pyodide.runPython(`
from pytgs.tgs import tgs_fit
from pytgs.functions import tgs_function
from pytgs.signal_process import process_signal
from pytgs.fft import fft as tgs_fft
from pytgs.lorentzian import lorentzian_fit
`);

  log("info", "Pyodide ready.");
}

async function ensureReady() {
  if (!initPromise) initPromise = initialize();
  return initPromise;
}

function writeSignalFile(pathStr, arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  pyodide.FS.writeFile(pathStr, bytes);
}

async function runFit({ jobId, fileIdx, fileId, posBytes, negBytes, config }) {
  const dir = `/tmp/job-${jobId}-${fileIdx}`;
  try {
    pyodide.FS.mkdirTree(dir);
  } catch (_) {}

  const posPath = `${dir}/${fileId}-POS-${fileIdx}.txt`;
  const negPath = `${dir}/${fileId}-NEG-${fileIdx}.txt`;
  writeSignalFile(posPath, posBytes);
  writeSignalFile(negPath, negBytes);

  pyodide.globals.set("_web_config_json", JSON.stringify(config));
  pyodide.globals.set("_web_pos", posPath);
  pyodide.globals.set("_web_neg", negPath);

  pyodide.runPython(`
import json
import numpy as np
from pytgs.tgs import tgs_fit
from pytgs.functions import tgs_function

_cfg = json.loads(_web_config_json)

_tgs_cfg = _cfg.get("tgs", {})
_grating   = _tgs_cfg.get("grating_spacing", 3.5276)
_sigprop   = _tgs_cfg.get("signal_proportion", 1.0)
_maxfev    = _tgs_cfg.get("maxfev", 1000000)

_result = tgs_fit(
    _cfg, _web_pos, _web_neg,
    grating_spacing=_grating,
    signal_proportion=_sigprop,
    maxfev=_maxfev,
)
(start_idx, start_time, grating_spacing,
 A, A_err, B, B_err, C, C_err,
 alpha, alpha_err, beta, beta_err,
 theta, theta_err, tau, tau_err,
 f, f_err, signal, fft_full, lorentzian_curve) = _result

# Reconstruct the fit curves at the fit-domain time points so the UI can
# draw them as Plotly traces.
functional_function, thermal_function = tgs_function(start_time, grating_spacing)
_params = (float(A), float(B), float(C), float(alpha), float(beta),
           float(theta), float(tau), float(f))

_end_idx = min(len(signal), int(len(signal) * _sigprop) + int(start_idx))

_time_full = signal[:, 0]
_amp_full  = signal[:, 1]
_time_fit  = signal[int(start_idx):_end_idx, 0]
_y_func    = functional_function(_time_fit, *_params)
_y_therm   = thermal_function(_time_fit, *_params)

# Lorentzian curve x-axis: 500 points across the configured frequency bounds.
_bounds    = _cfg.get("lorentzian", {}).get("frequency_bounds", [0.1, 0.9])
_lor_x     = np.linspace(float(_bounds[0]), float(_bounds[1]), len(lorentzian_curve))

_out = {
    "startIdx": int(start_idx),
    "startTime": float(start_time),
    "gratingSpacing": float(grating_spacing),
    "params": {
        "A":     {"value": float(A),     "err": float(A_err),     "unit": "Wm^-2"},
        "B":     {"value": float(B),     "err": float(B_err),     "unit": "Wm^-2"},
        "C":     {"value": float(C),     "err": float(C_err),     "unit": "Wm^-2"},
        "alpha": {"value": float(alpha), "err": float(alpha_err), "unit": "m^2s^-1"},
        "beta":  {"value": float(beta),  "err": float(beta_err),  "unit": "s^0.5"},
        "theta": {"value": float(theta), "err": float(theta_err), "unit": "rad"},
        "tau":   {"value": float(tau),   "err": float(tau_err),   "unit": "s"},
        "f":     {"value": float(f),     "err": float(f_err),     "unit": "Hz"},
    },
    "traces": {
        "tgs": {
            "timeFull":     _time_full.tolist(),
            "ampFull":      _amp_full.tolist(),
            "timeFit":      _time_fit.tolist(),
            "yFunctional":  _y_func.tolist(),
            "yThermal":     _y_therm.tolist(),
        },
        "fftLorentzian": {
            "frequency":    fft_full[:, 0].tolist(),
            "amplitude":    fft_full[:, 1].tolist(),
            "curveX":       _lor_x.tolist(),
            "curveY":       [float(v) for v in lorentzian_curve],
            "bounds":       [float(_bounds[0]), float(_bounds[1])],
        },
    },
    "signalLength": int(len(signal)),
}

_out_json = json.dumps(_out)
`);

  const outJson = pyodide.globals.get("_out_json");
  const out = JSON.parse(outJson);

  try {
    pyodide.runPython(`
import shutil
shutil.rmtree("${dir}", ignore_errors=True)
`);
  } catch (_) {}

  return out;
}

async function runCalibrate({ posBytes, negBytes, config, soundSpeed }) {
  const dir = `/tmp/calib-${Date.now()}`;
  try {
    pyodide.FS.mkdirTree(dir);
  } catch (_) {}
  const posPath = `${dir}/CAL-POS-1.txt`;
  const negPath = `${dir}/CAL-NEG-1.txt`;
  writeSignalFile(posPath, posBytes);
  writeSignalFile(negPath, negBytes);

  pyodide.globals.set("_web_config_json", JSON.stringify(config));
  pyodide.globals.set("_web_pos", posPath);
  pyodide.globals.set("_web_neg", negPath);
  pyodide.globals.set("_web_sound_speed", soundSpeed);

  pyodide.runPython(`
import json
import numpy as np
from pytgs.signal_process import process_signal
from pytgs.fft import fft as tgs_fft
from pytgs.lorentzian import lorentzian_fit

_cfg = json.loads(_web_config_json)

_nominal = 3.5276
signal, max_time, start_time, start_idx = process_signal(
    _web_pos, _web_neg, _nominal, **_cfg["signal_process"]
)

_t = signal[:, 0]
_y = signal[:, 1] - float(np.mean(signal[:, 1]))
_saw = np.column_stack([_t, _y])
_fft_signal = tgs_fft(_saw, **_cfg["fft"])

(f, f_err, fwhm, tau, snr, freq_bounds,
 lorentzian_function, lorentzian_popt,
 fft_segment, fft_full, lorentzian_curve) = lorentzian_fit(
    _fft_signal, **_cfg["lorentzian"]
)

_spacing_um = (_web_sound_speed / f) * 1e6

_out = {
    "frequency": float(f),
    "frequencyErr": float(f_err),
    "gratingSpacing": float(_spacing_um),
    "snr": float(snr),
    "fwhm": float(fwhm),
    "tau": float(tau),
}
_out_json = json.dumps(_out)
`);

  const outJson = pyodide.globals.get("_out_json");
  const out = JSON.parse(outJson);

  try {
    pyodide.runPython(`
import shutil
shutil.rmtree("${dir}", ignore_errors=True)
`);
  } catch (_) {}

  return out;
}

self.onmessage = async (e) => {
  const msg = e.data;
  const { id, type } = msg;
  try {
    switch (type) {
      case "init": {
        await ensureReady();
        self.postMessage({ id, type: "ready" });
        break;
      }
      case "fit": {
        await ensureReady();
        self.postMessage({ id, type: "progress", stage: "running" });
        const result = await runFit(msg);
        self.postMessage({ id, type: "result", result });
        break;
      }
      case "calibrate": {
        await ensureReady();
        self.postMessage({ id, type: "progress", stage: "running" });
        const result = await runCalibrate(msg);
        self.postMessage({ id, type: "result", result });
        break;
      }
      default:
        self.postMessage({
          id,
          type: "error",
          message: `Unknown request type: ${type}`,
        });
    }
  } catch (err) {
    const message = stringifyError(err);
    log("error", `Worker handler failed (${type}): ${message}`);
    self.postMessage({ id, type: "error", message });
  }
};

self.onerror = (ev) => {
  const message = stringifyError(ev?.error ?? ev?.message ?? ev);
  self.postMessage({ type: "log", level: "error", message: `Worker error: ${message}` });
};
self.onunhandledrejection = (ev) => {
  const message = stringifyError(ev?.reason ?? ev);
  self.postMessage({
    type: "log",
    level: "error",
    message: `Worker unhandled rejection: ${message}`,
  });
};
