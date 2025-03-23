import json
from typing import Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from src.analysis.tgs import tgs_fit
from src.core.utils import get_num_signals, get_file_prefix
from src.core.path import Paths

class TGSAnalyzer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        base_path = Path(config['path'])
        self.paths = Paths(
            data_dir=base_path,
            figure_dir=base_path / 'figures',
            fit_dir=base_path / 'fit',
            fit_path=base_path / 'fit' / 'fit.csv',
            signal_path=base_path / 'fit' / 'signal.json',
        )
        self.paths.fit_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figure_dir.mkdir(parents=True, exist_ok=True)
        
        study_signals = get_num_signals(self.paths.data_dir)
        if config['study_names'] is not None:
            study_signals = {study: max_idx 
                            for study, max_idx in study_signals.items() 
                            if study in config['study_names']}

        if config['idxs'] is not None:
            self.idxs = [(study, idx) for study in study_signals.keys() for idx in config['idxs'] if idx <= study_signals[study]]
        else:
            self.idxs = [(study, idx) 
                         for study, max_idx in study_signals.items() 
                         for idx in range(1, max_idx + 1)]

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def fit_signal(self, file_idx: int, pos_file: str, neg_file: str) -> Tuple[pd.DataFrame, List[List[float]], List[List[float]]]:
        (start_idx, start_time, grating_spacing, 
         A, A_err, B, B_err, C, C_err, 
         alpha, alpha_err, beta, beta_err, 
         theta, theta_err, tau, tau_err, 
         f, f_err, signal) = tgs_fit(self.config, self.paths, file_idx, pos_file, neg_file, **self.config['tgs'])

        params = {
            'A': (A, A_err, 'Wm^-2'),
            'B': (B, B_err, 'Wm^-2'),
            'C': (C, C_err, 'Wm^-2'),
            'alpha': (alpha, alpha_err, 'm^2s^-1'),
            'beta': (beta, beta_err, 's^0.5'),
            'theta': (theta, theta_err, 'rad'),
            'tau': (tau, tau_err, 's'),
            'f': (f, f_err, 'Hz'),
        }

        data = {
            'run_name': Path(pos_file).name,
            'start_idx': start_idx,
            'start_time': start_time,
            'grating_spacing[µm]': grating_spacing,
            **{f'{name}[{unit}]': value for name, (value, _, unit) in params.items()},
            **{f'{name}_err[{unit}]': error for name, (_, error, unit) in params.items()},
        }

        return pd.DataFrame([data]), signal.tolist()

    def fit(self, show: bool = True) -> None:
        self.logger.setLevel(logging.INFO if show else logging.WARNING)
        
        fit_data = pd.DataFrame()
        signals = []
        fails = []

        if self.idxs is None:
            study_signals = get_num_signals(self.paths.data_dir)
            self.idxs = [(study, idx) 
                         for study, max_idx in study_signals.items() 
                         for idx in range(1, max_idx + 1)]

        for study_name, i in self.idxs:
            self.logger.info(f"Analyzing {study_name} signal {i}")
            if not (file_prefix := get_file_prefix(self.paths.data_dir, i, study_name)):
                msg = f"Could not find file prefix for signal {i} in study {study_name}"
                self.logger.warning(msg)
                fails.append((study_name, i, msg))
                continue

            pos_file = self.paths.data_dir / f'{file_prefix}-{study_name}-POS-{i}.txt'
            neg_file = self.paths.data_dir / f'{file_prefix}-{study_name}-NEG-{i}.txt'

            try:
                df, signal = self.fit_signal(i, pos_file, neg_file)
                signals.append(signal)
                fit_data = pd.concat([fit_data, df], ignore_index=True)
            except Exception as e:
                msg = f"Error fitting signal {i} from study {study_name}: {str(e)}"
                self.logger.warning(msg)
                fails.append((study_name, i, msg))
                continue

        fit_data.to_csv(self.paths.fit_path, index=False)
        with open(self.paths.signal_path, 'w') as f:
            json.dump(signals, f)
    
        if show:
            self.fit_summary(fails)

    def fit_summary(self, fails: List[Tuple[str, int, str]] = None) -> None:
        if not self.paths.fit_path.exists():
            self.logger.warning("No fit data found. Please run fit() first.")
            return

        fit_data = pd.read_csv(self.paths.fit_path)
        param_cols = [col for col in fit_data.columns 
                     if any(param in col for param in ['A[', 'B[', 'C[', 'alpha[', 'beta[', 'theta[', 'tau[', 'f['])
                     and not 'err' in col]
        
        summary = []
        for param in param_cols:
            try:
                values = fit_data[param].values
                param_base = param.split('[')[0]
                unit = param.split('[')[1]
                error_col = f"{param_base}_err[{unit}"
                errors = fit_data[error_col].values
                
                weights = 1 / (errors ** 2)
                if np.all(weights == 0) or np.any(~np.isfinite(weights)):
                    self.logger.warning(f"Warning: Invalid weights for parameter {param}. Using unweighted statistics.")
                    weighted_mean = np.mean(values)
                    weighted_std = np.std(values)
                else:
                    weighted_mean = np.average(values, weights=weights)
                    weighted_std = np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights))
                
                summary.append({
                    'Parameter': param,
                    'Mean': weighted_mean,
                    'Std': weighted_std,
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Relative Error (%)': np.mean(errors / np.abs(values)) * 100 if np.any(values != 0) else np.inf
                })
            except Exception as e:
                self.logger.warning(f"Error processing parameter {param}: {str(e)}")
                summary.append({
                    'Parameter': param,
                    'Mean': np.nan,
                    'Std': np.nan,
                    'Min': np.nan,
                    'Max': np.nan,
                    'Relative Error (%)': np.nan
                })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.set_index('Parameter')
        
        self.logger.info("\nFit Summary:")
        self.logger.info("-" * 80)
        self.logger.info(f"Total signals attempted: {len(fit_data) + len(fails if fails else [])}")
        self.logger.info(f"Successful fits: {len(fit_data)}")
        self.logger.info(f"Failed fits: {len(fails) if fails else 0}")
        
        if fails:
            self.logger.info("\nFailed Fits:")
            for study, idx, error in fails:
                self.logger.info(f"- Study: {study}, Signal: {idx}")
                self.logger.info(f"  Error: {error}")
        
        if 'grating_spacing[µm]' in fit_data.columns:
            self.logger.info(f"\nGrating spacing: {fit_data['grating_spacing[µm]'].iloc[0]:.4f} µm")
        self.logger.info("\nParameter Statistics:")
        self.logger.info(summary_df.round(6).to_string())
        
        summary_path = self.paths.fit_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Fit Summary\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Total signals attempted: {len(fit_data) + len(fails if fails else [])}\n")
            f.write(f"Successful fits: {len(fit_data)}\n")
            f.write(f"Failed fits: {len(fails) if fails else 0}\n\n")
            
            if fails:
                f.write("Failed Fits:\n")
                for study, idx, error in fails:
                    f.write(f"- Study: {study}, Signal: {idx}\n")
                    f.write(f"  Error: {error}\n")
                f.write("\n")
            
            if 'grating_spacing[µm]' in fit_data.columns:
                f.write(f"Grating spacing: {fit_data['grating_spacing[µm]'].iloc[0]:.4f} µm\n")
            f.write("\nParameter Statistics:\n")
            f.write(f"{'-' * 80}\n")
            f.write(summary_df.round(6).to_string())
        
