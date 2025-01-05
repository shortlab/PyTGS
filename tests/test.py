import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import pytest

from tests.test_utils import generate_signal, save
from src.core.fit import TGSAnalyzer

def typical_signal():
    params = {
        'A': 0.0005,
        'B': 0.05,
        'C': 0,
        'alpha': 5e-6,
        'beta': 0.05,
        'theta': np.pi/4,
        'tau': 50e-9,
        'f': 5e8,
        'grating_spacing': 3.5
    }
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    pos_signal = generate_signal(params, sign='POS')
    neg_signal = generate_signal(params, sign='NEG')
    return pos_signal, neg_signal, params, config

def high_thermal_diffusivity():
    params = {
        'A': 0.007,
        'B': 0.05,
        'C': 0,
        'alpha': 4e-5,
        'beta': 0.005,
        'theta': np.pi/4,
        'tau': 50e-9,
        'f': 5e8,
        'grating_spacing': 3.5
    }
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['tgs']['signal_proportion'] = 0.4
    
    pos_signal = generate_signal(params, sign='POS', noise_level=0.001)
    neg_signal = generate_signal(params, sign='NEG', noise_level=0.001)
    return pos_signal, neg_signal, params, config

def strong_acoustic_component():
    params = {
        'A': 0.0005,
        'B': 0.08,
        'C': 0,
        'alpha': 5e-6,
        'beta': 0.05,
        'theta': np.pi/4,
        'tau': 50e-9,
        'f': 7e8,
        'grating_spacing': 3.5
    }
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    pos_signal = generate_signal(params, sign='POS')
    neg_signal = generate_signal(params, sign='NEG')
    return pos_signal, neg_signal, params, config

def extra_noisy():
    params = {
        'A': 0.0005,
        'B': 0.05,
        'C': 0,
        'alpha': 5e-6,
        'beta': 0.05,
        'theta': np.pi/4,
        'tau': 50e-9,
        'f': 5e8,
        'grating_spacing': 3.5
    }
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    pos_signal = generate_signal(params, sign='POS', noise_level=0.1)
    neg_signal = generate_signal(params, sign='NEG', noise_level=0.1)
    return pos_signal, neg_signal, params, config

@pytest.mark.parametrize("test_case,gen_func", [
    (1, typical_signal),
    (2, high_thermal_diffusivity),
    (3, strong_acoustic_component),
    (4, extra_noisy)
])
def test(test_case, gen_func):
    pos_signal, neg_signal, params, config = gen_func()

    config['path'] = f'tests/test-{test_case}'
    config['study_names'] = ['synthetic']
    config['tgs']['grating_spacing'] = params['grating_spacing']
    config['signal_process']['initial_samples'] = -50

    save([(test_case, (pos_signal, neg_signal, params, config))])
    
    alpha_errors = []
    f_errors = []
    for _ in range(3):
        analyzer = TGSAnalyzer(config)
        analyzer.fit(show=False)

        fit_results = pd.read_csv(analyzer.paths.fit_path)
        true_params = json.load(open(analyzer.paths.data_dir / f'test-{test_case}-params.json'))
        
        fitted_alpha = fit_results.iloc[0]['alpha[m^2s^-1]']
        fitted_f = fit_results.iloc[0]['f[Hz]']
        
        alpha_error = abs(fitted_alpha - true_params['alpha']) / true_params['alpha'] * 100
        f_error = abs(fitted_f - true_params['f']) / true_params['f'] * 100
        
        alpha_errors.append(alpha_error)
        f_errors.append(f_error)
    
    avg_alpha_error = float(np.mean(alpha_errors))
    avg_f_error = float(np.mean(f_errors))

    assert avg_alpha_error < 10, f"Average alpha error {avg_alpha_error:.2f}% > 10%"
    assert avg_f_error < 10, f"Average f error {avg_f_error:.2f}% > 10%"