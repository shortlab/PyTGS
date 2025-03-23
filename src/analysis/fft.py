import numpy as np
from scipy.signal import periodogram, savgol_filter, windows

NOISE_CUTOFF_POINTS = 1000

def fft(saw_signal: np.ndarray, signal_proportion: float = 0.9, use_derivative: bool = True, analysis_type: str = 'psd') -> np.ndarray:
    """
    Generates a filtered Fast Fourier Transform of SAW signal.
    
    Parameters:
        saw_signal (np.ndarray): SAW signal data array of shape (N, 2) containing frequency and amplitude
        signal_proportion (float, optional): proportion of signal to include in fit
        duplicate_signal (bool, optional): whether to duplicate the signal to artificially increase frequency resolution
        use_derivative (bool, optional): whether to use the derivative of the signal
        noise_cutoff_points (int, optional): number of points zero out at start/end of spectrum to remove noise
        analysis_type (str, optional): whether to output power spectral density ('psd') or fast fourier transform ('fft')
        
    Returns:
        transformed_signal (np.ndarray): transformed signal array of shape (M, 2) where:
            - First column contains frequency values in Hz
            - Second column contains corresponding amplitudes
            - M is the number of frequency points after transformation
            - If analysis_type='psd', amplitudes represent power spectral density
            - If analysis_type='fft', amplitudes represent Fourier coefficients
    """
    N, _ = saw_signal.shape
    M = int(np.ceil(N * signal_proportion))

    saw_signal = saw_signal[:M]
    saw_signal[:, 1] /= np.max(saw_signal[:, 1])

    # if duplicate_signal:
    #     mirror_signal = np.flipud(saw_signal[1:])
    #     mirror_signal[:, 0] = saw_signal[-1, 0] + (mirror_signal[:, 0] - saw_signal[0, 0])
    #     saw_signal = np.vstack((saw_signal, mirror_signal))

    time = saw_signal[:, 0]
    amplitude = saw_signal[:, 1]
    t_step = time[-1] - time[-2]

    if use_derivative:
        derivative = np.diff(amplitude) / t_step
        derivative /= np.max(derivative)
        saw_signal = np.vstack((time[:-1], derivative)).T

    num_points = len(saw_signal)
    fs = num_points / (time[-1] - time[0])
    pad_size = 2 ** 18 - num_points - 2
    pad_time = np.arange(time[-1], time[-1] + pad_size * t_step, t_step)
    pad_amplitude = np.full(pad_size, 0)

    if len(pad_time) != len(pad_amplitude):
        min_pad_length = min(len(pad_time), len(pad_amplitude))
        pad_time = pad_time[:min_pad_length]
        pad_amplitude = pad_amplitude[:min_pad_length]
    
    pad_signal = np.vstack((
        np.hstack((time, pad_time)),
        np.hstack((amplitude, pad_amplitude))
    )).T

    nfft = len(pad_signal)
    frequencies, power_spectral_density = periodogram(
        pad_signal[:, 1],
        fs=fs,
        window=windows.hamming(nfft),
        nfft=nfft
    )

    npsd = int(np.ceil(len(power_spectral_density) * 0.2)) - 6 * NOISE_CUTOFF_POINTS
    power_spectral_density[:NOISE_CUTOFF_POINTS] = 0
    power_spectral_density[npsd:] = 0
    power_spectral_density /= np.max(power_spectral_density)

    if analysis_type == 'psd':
        amplitudes = power_spectral_density[:-1]
    elif analysis_type == 'fft':
        amplitudes = np.sqrt(power_spectral_density[:-1])

    filtered_amplitudes = savgol_filter(amplitudes, window_length=201, polyorder=5)
    transformed_signal = np.vstack((frequencies[:-1], filtered_amplitudes)).T

    return transformed_signal
