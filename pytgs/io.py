import re
from pathlib import Path

import numpy as np

def read_data(file_path: str) -> np.ndarray:
    """
    Read a TGS signal data file and return [time, amplitude] as a numpy array.

    Parameters:
        file_path (str): path to the TGS data file
    Returns:
        np.ndarray: array of shape (N, 2) containing [time, amplitude]
    """
    time_data = []
    amplitude_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().lower().startswith('time') and 'amplitude' in line.lower():
                break
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t, a = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            time_data.append(t)
            amplitude_data.append(a)

    if not time_data:
        raise ValueError(f"No numeric samples found in {file_path}")

    return np.column_stack((time_data, amplitude_data))

def get_num_signals(path: Path) -> dict[str, int]:
    """
    Get the number of positive signal files for each study in the given path.

    Parameters:
        path (str): path to the directory containing the signal files
    Returns:
        dict[str, int]: dictionary mapping study names to their maximum signal index
    """
    pattern = re.compile(r'.*-([\w.]+)-((?:POS|NEG)-\d+)\.txt$')
    
    study_indices = {}
    for filename in path.iterdir():
        if match := pattern.search(filename.name):
            study_name = match.group(1)
            index = int(match.group(2).split('-')[-1])
            
            if study_name not in study_indices:
                study_indices[study_name] = set()
            study_indices[study_name].add(index)
            
    return {study: max(indices) for study, indices in study_indices.items()}

def get_file_prefix(path: Path, i: int, study_name: str) -> str:
    """
    Get the file prefix of the signal file with the given index and study name.

    Parameters:
        path (str): path to the directory containing the signal files
        i (int): index of the signal file
        study_name (str): name of the study
    Returns:
        str: file prefix
    """
    pattern = re.compile(
        rf'(.+)-{re.escape(study_name)}-((?:POS|NEG)-{i})\.txt'
    )
    for filename in path.iterdir():
        if match := pattern.match(filename.name):
            return match.group(1)
    return None
