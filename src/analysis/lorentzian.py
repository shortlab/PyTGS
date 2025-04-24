from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from src.core.path import Paths
from src.analysis.functions import lorentzian_function, super_lorentzian_function
from src.core.plots import plot_fft_lorentzian


def lorentzian_fit(config: dict, paths: Paths, file_idx: int, fft: np.ndarray, signal_proportion: float = 1.0, frequency_bounds: List[Union[float, float]] = [0.1, 0.9], dc_filter_range: List[Union[int, int]] = [0, 12000], bimodal_fit: bool = False, use_super_lorentzian: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit Lorentzian peak to FFT signal.

    This function performs either single or bimodal Lorentzian peak fitting on FFT data.
    It includes DC filtering, signal normalization, and optional partial signal fitting
    based on a proportion of the peak height.

    Parameters:
        config (dict): configuration dictionary
        paths (Paths): paths to data, figures, and fit files
        file_idx (int): file index
        fft (np.ndarray): FFT signal array of shape (N, 2) containing frequency and amplitude
        signal_proportion (float, optional): proportion of signal to include in fit
        frequency_range (List[float], optional): [min, max] frequency bounds for fitting [GHz]
        dc_filter_range (List[int], optional): [start, end] indices for DC filtering
        bimodal_fit (bool, optional): whether to perform bimodal peak fitting
        use_super_lorentzian (bool, optional): whether to use super-Lorentzian for flat-top peaks

    Returns:
        Tuple: contains the following elements:
            - peak (np.ndarray): peak frequency [Hz]
            - peak_error (np.ndarray): peak frequency error [Hz]
            - fwhm (np.ndarray): full width at half maximum [Hz]
            - tau (np.ndarray): time constant [s]
            - snr (float): signal-to-noise ratio [dB]
            - frequency_bounds (List[float, float]): frequency bounds for fitting [GHz]
            - fit_function (function): Fitting function used
            - popt (np.ndarray): optimized fit parameters
    """
    start, end = dc_filter_range
    fft[:, 0] = fft[:, 0] / 1e9
    fft[:start, 1] = 0

    max_value = np.max(fft[start:, 1])
    peak_idx = np.argmax(fft[start:, 1]) 
    peak_loc = fft[peak_idx, 0]

    fft[:, 1] /= max_value

    if signal_proportion != 1.0:
        threshold = signal_proportion * fft[peak_idx + start, 1]
        neg_idx = peak_idx + start
        while neg_idx > start and fft[neg_idx, 1] > threshold:
            neg_idx -= 1
        pos_idx = peak_idx + start
        while pos_idx < len(fft) and fft[pos_idx, 1] > threshold:
            pos_idx += 1
    else:
        neg_idx = start
        pos_idx = len(fft)

    min_freq, max_freq = frequency_bounds
    
    if use_super_lorentzian:
        fit_function = super_lorentzian_function
        initial_guess = [1e-2, peak_loc, 0.05, 0, 0.5]
        lower_bounds = [0, min_freq, 1e-3, 0, 0.1]
        upper_bounds = [1, max_freq, 0.2, 1, 0.9]
    else:
        fit_function = lorentzian_function
        initial_guess = [1e-4, peak_loc, 1e-2, 0]
        lower_bounds = [0, min_freq, 1e-3, 0]
        upper_bounds = [1, max_freq, 0.05, 1]
    
    bounds = (lower_bounds, upper_bounds)

    popt, pcov = curve_fit(fit_function, fft[neg_idx:pos_idx, 0], fft[neg_idx:pos_idx, 1], 
                          p0=initial_guess, bounds=bounds)
    
    if use_super_lorentzian:
        _, x0, W, _, _ = popt
        _, x0_error, _, _, _ = np.sqrt(np.diag(pcov))
    else:
        _, x0, W, _ = popt
        _, x0_error, _, _ = np.sqrt(np.diag(pcov))
    
    saw_frequency = x0 * 1e9
    saw_frequency_error = x0_error * 1e9
    fwhm = 2 * W * 1e9
    tau = 1 / (np.pi * fwhm)

    if bimodal_fit:
        bimodal_start = round(0.1 * peak_idx)
        bimodal_end = round(0.75 * peak_idx)

        fft2 = fft[:, 1] - fit_function(fft[:, 0], *popt)

        peak_idx2 = np.argmax(fft2[bimodal_start:bimodal_end]) + bimodal_start
        peak_loc2 = fft[peak_idx2, 0]

        initial_guess2 = [1e-4, peak_loc2, 0.01, 0]
        popt2, pcov2 = curve_fit(fit_function, fft[:, 0], fft2, p0=initial_guess2, bounds=bounds)
        _, x02, W2, _ = popt2
        _, x02_error, _, _ = np.sqrt(np.diag(pcov2))
        saw_frequency2 = x02 * 1e9
        saw_frequency_error2 = x02_error * 1e9
        fwhm2 = 2 * W2 * 1e9
        tau2 = 1 / (np.pi * fwhm2)

        saw_frequency = np.array([saw_frequency, saw_frequency2])
        saw_frequency_error = np.array([saw_frequency_error, saw_frequency_error2])
        fwhm = np.array([fwhm, fwhm2])
        tau = np.array([tau, tau2])
        
    fft_noise = np.column_stack((fft[:, 0], fft[:, 1] - fit_function(fft[:, 0], *popt)))
    signal_power = np.mean(fft[:, 1] ** 2)
    noise_power = np.mean(fft_noise[:, 1] ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    if config['plot']['fft_lorentzian']:
        plot_fft_lorentzian(paths, file_idx, fft[neg_idx:pos_idx], frequency_bounds, fit_function, popt)

    return saw_frequency, saw_frequency_error, fwhm, tau, snr, frequency_bounds, fit_function, popt