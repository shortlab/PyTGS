from pathlib import Path
from dataclasses import dataclass

@dataclass
class Paths:
    data_dir: Path
    figure_dir: Path
    fit_dir: Path
    fit_path: Path
    signal_path: Path