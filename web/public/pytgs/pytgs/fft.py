import numpy as np
from scipy.signal import periodogram, savgol_filter, windows

NOISE_CUTOFF_POINTS = 1000
NFFT_TARGET = 2 ** 18
SAVGOL_WINDOW = 201
SAVGOL_POLYORDER = 5


def fft(
    saw_signal: np.ndarray,
    signal_proportion: float = 1.0,
    analysis_type: str = 'psd',
    use_derivative: bool = False,
) -> np.ndarray:
    """
    Generates a filtered Fast Fourier Transform of SAW signal.
    Parameters:
        saw_signal: (N, 2) array of [time s, amplitude].
        signal_proportion: fraction of the signal to keep (0, 1].
        analysis_type: 'psd' for power spectral density or 'fft' for |FFT|.
        use_derivative: FFT the time derivative rather than the raw signal.

    Returns:
        (M, 2) array of [frequency Hz, spectrum amplitude].
    """
    if analysis_type not in ('psd', 'fft'):
        raise ValueError(
            f"analysis_type must be 'psd' or 'fft', got {analysis_type!r}"
        )

    N = len(saw_signal)
    if N < 2:
        raise ValueError(f"saw_signal must have at least 2 samples, got {N}")
    M = max(2, int(np.ceil(N * signal_proportion)))

    time = np.asarray(saw_signal[:M, 0], dtype=float)
    amplitude = np.asarray(saw_signal[:M, 1], dtype=float)

    amp_max = float(np.max(np.abs(amplitude)))
    if amp_max > 0:
        amplitude = amplitude / amp_max

    t_step = float(time[-1] - time[-2])
    if t_step <= 0:
        raise ValueError("saw_signal time column must be strictly increasing")
    fs = 1.0 / t_step

    if use_derivative:
        data = np.gradient(amplitude, t_step)
    else:
        data = amplitude

    num_points = len(data)
    windowed = data * windows.hamming(num_points)

    nfft = max(NFFT_TARGET, num_points)
    padded = np.concatenate([windowed, np.zeros(nfft - num_points)])

    frequencies, psd = periodogram(padded, fs=fs, window='boxcar', nfft=nfft)

    cutoff = min(NOISE_CUTOFF_POINTS, len(psd))
    psd[:cutoff] = 0

    psd_max = float(np.max(psd))
    if psd_max > 0:
        psd = psd / psd_max

    amplitudes = psd if analysis_type == 'psd' else np.sqrt(psd)

    window_length = min(SAVGOL_WINDOW, len(amplitudes) - (1 if len(amplitudes) % 2 == 0 else 0))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length > SAVGOL_POLYORDER:
        amplitudes = savgol_filter(
            amplitudes, window_length=window_length, polyorder=SAVGOL_POLYORDER
        )

    return np.column_stack((frequencies, amplitudes))
