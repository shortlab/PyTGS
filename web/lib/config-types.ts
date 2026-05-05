// Types mirror config.yaml / the TGSAnalyzer dict keys exactly.

export type HeterodyneMode = "di-homodyne" | "mono-homodyne";
export type FftAnalysisType = "fft" | "psd";

export interface BaselineCorrectionConfig {
  enabled: boolean;
  pos: string | null;
  neg: string | null;
}

export interface SignalProcessConfig {
  heterodyne: HeterodyneMode;
  null_point: 1 | 2 | 3 | 4;
  initial_samples: number;
  baseline_correction: BaselineCorrectionConfig;
}

export interface FftConfig {
  signal_proportion: number;
  analysis_type: FftAnalysisType;
}

export interface LorentzianConfig {
  signal_proportion: number;
  frequency_bounds: [number, number];
  dc_filter_range: [number, number];
  bimodal_fit: boolean;
  use_skewed_super_lorentzian: boolean;
}

export interface TgsFitConfig {
  grating_spacing: number;
  signal_proportion: number;
  maxfev: number;
}

export interface PyTGSConfig {
  path: string;
  study_names: string[] | null;
  idxs: unknown | null;
  signal_process: SignalProcessConfig;
  fft: FftConfig;
  lorentzian: LorentzianConfig;
  tgs: TgsFitConfig;
}

// Fit parameter names the API returns.
export type FitParamName =
  | "A"
  | "B"
  | "C"
  | "alpha"
  | "beta"
  | "theta"
  | "tau"
  | "f";

export interface FitParam {
  value: number;
  err: number;
  unit: string;
}

export type PlotKind = "tgs" | "fft-lorentzian";

export interface TgsTrace {
  timeFull: number[];
  ampFull: number[];
  timeFit: number[];
  yFunctional: number[];
  yThermal: number[];
}

export interface FftLorentzianTrace {
  frequency: number[];
  amplitude: number[];
  curveX: number[];
  curveY: number[];
  bounds: [number, number];
}

export interface FitResult {
  startIdx: number;
  startTime: number;
  gratingSpacing: number;
  params: Record<FitParamName, FitParam>;
  traces: {
    tgs: TgsTrace;
    fftLorentzian: FftLorentzianTrace;
  };
  signalLength: number;
}

export interface CalibrationResult {
  frequency: number;
  frequencyErr: number;
  gratingSpacing: number;
  snr: number;
  fwhm: number;
  tau: number;
}

export type LogLevel = "info" | "warning" | "error";

export interface WorkerLog {
  level: LogLevel;
  message: string;
  timestamp: number;
}
