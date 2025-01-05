import json
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

from src.analysis.functions import tgs_function

def generate_signal(params, time_range=(1e-12, 2e-7), num_points=4000, noise_level=0.05, sign='POS'):
    np.random.seed(int(datetime.now().timestamp() * 1e6) % (2**32 - 1))

    t = np.linspace(time_range[0], time_range[1], num_points)
    functional_fit, _ = tgs_function(start_time=2e-9, grating_spacing=params["grating_spacing"])
    signal = functional_fit(
        t,
        params["A"],
        params["B"],
        params["C"],
        params["alpha"],
        params["beta"],
        params["theta"],
        params["tau"],
        params["f"],
    )
    noise = np.random.normal(0, noise_level * np.std(signal), num_points)
    noisy_signal = signal + noise
    if sign == 'NEG':
        noisy_signal = -noisy_signal
    return np.column_stack((t, noisy_signal))

def save_signal(signal: np.ndarray, filepath: Path, metadata: dict) -> None:
    header = (
        f"Study Name\t{metadata['study_name']}\n"
        f"Sample Name\t{metadata['sample_name']}\n"
        f"Run Name\t{metadata['run_name']}\n"
        f"Operator\ttester\n"
        f"Date\t{metadata['date']}\n"
        f"Time\t{metadata['time']}\n"
        f"Sign\t{metadata['sign']}\n"
        f"Grating Spacing\t{metadata['grating_spacing']}um\n"
        f"Channel\t3\n"
        f"Number Traces\t10000\n"
        f"Files in Batch\t1\n"
        f"Batch Number\t{metadata['batch']}\n"
        f"dt\t50.000001E-12\n"
        f"time stamp (ms)\t12:00:00 PM\n"
        "\n"
        "Time\tAmplitude\n"
        ""
    )
    with open(filepath, 'w', newline='') as f:
        f.write(header)
        for time, amplitude in signal:
            f.write(f"{time:.6E}\t{amplitude:.6E}\n")


def save(test_cases, base_dir='tests'):
    data_dir = Path(base_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    for test_num, (pos_signal, neg_signal, params, config) in test_cases:
        test_dir = data_dir / f'test-{test_num}'
        test_dir.mkdir(parents=True, exist_ok=True)

        config_path = test_dir / f'test-{test_num}-config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)

        metadata = {
            'study_name': f'test-{test_num}',
            'sample_name': f'test-sample',
            'run_name': 'test-run',
            'date': current_date,
            'time': current_time,
            'grating_spacing': f"{params['grating_spacing']:05.2f}",
            'batch': 1
        }

        metadata['sign'] = 'POS'
        pos_path = test_dir / f'test-{test_num}-synthetic-POS-1.txt'
        save_signal(pos_signal, pos_path, metadata)
        
        metadata['sign'] = 'NEG'
        neg_path = test_dir / f'test-{test_num}-synthetic-NEG-1.txt'
        save_signal(neg_signal, neg_path, metadata)

        with open(test_dir / f'test-{test_num}-params.json', 'w') as f:
            json.dump(params, f, indent=4)
