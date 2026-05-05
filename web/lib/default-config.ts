import type { PyTGSConfig } from "./config-types";

// Mirrors config.yaml at the repo root.
export const DEFAULT_CONFIG: PyTGSConfig = {
  path: "examples",
  study_names: ["cooldown"],
  idxs: null,
  signal_process: {
    heterodyne: "di-homodyne",
    null_point: 2,
    initial_samples: 50,
    baseline_correction: {
      enabled: false,
      pos: null,
      neg: null,
    },
  },
  fft: {
    signal_proportion: 1.0,
    analysis_type: "fft",
  },
  lorentzian: {
    signal_proportion: 1,
    frequency_bounds: [0.1, 0.9],
    dc_filter_range: [0, 12000],
    bimodal_fit: false,
    use_skewed_super_lorentzian: false,
  },
  tgs: {
    grating_spacing: 3.5276,
    signal_proportion: 1,
    maxfev: 1000000,
  },
};

// A known-good SAW sound speed for the standard calibration material
// (pinned in the Tkinter GUI as 2665.9 m/s).
export const DEFAULT_CALIB_SOUND_SPEED = 2665.9;
