# PyTGS

A Python tool for analyzing transient grating spectroscopy (TGS) signals. Process raw signals and extract thermo-mechanical properties through automated analyses and curve fitting.

## Prerequisites

- Python 3.8 - 3.12 ([download](https://www.python.org/downloads/))
- Git ([download](https://git-scm.com/downloads))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shortlab/PyTGS.git
   cd PyTGS
   ```

2. Run the automated setup script:
   
   **On Unix/MacOS:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   **On Windows:**
   ```bash
   bash setup.sh
   ```
   
   Note: On Windows, you'll need Git Bash or a similar bash shell. If you don't have one use the manual installation below.

   **Manual Installation (Windows/Unix/MacOS):**
   ```bash
   python -m venv PyTGS-venv
   # On Windows:
   PyTGS-venv\Scripts\activate
   # On Unix/MacOS:
   source PyTGS-venv/bin/activate
   
   pip install --upgrade pip
   pip install -e .  # or pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment:
   
   **On Unix/MacOS:**
   ```bash
   source PyTGS-venv/bin/activate
   ```
   
   **On Windows:**
   ```bash
   PyTGS-venv\Scripts\activate
   ```

2. Edit `config.yaml` with your data path and desired fitting parameters.

3. Run the analysis:
   ```bash
   python main.py
   ```

   Fitting results and figures will be saved in `fit/` and `figures/` directories, respectively.
   You can view example input/output files in the `example/` directory.

## Testing

The package includes tests that validate correctness of the analysis pipeline using synthetic TGS signals. 

To run the tests, use the following command:
```bash
pytest tests/test.py -v
```