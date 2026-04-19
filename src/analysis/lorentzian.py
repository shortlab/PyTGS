from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from src.core.path import Paths
from src.analysis.functions import lorentzian_function, skewed_super_lorentzian_function
from src.core.plots import plot_fft_lorentzian

def lorentzian_fit(
    config: dict,
    paths: Paths,
    file_id: int,
    fft: np.ndarray,
    signal_proportion: float = 1.0,
    frequency_bounds: List[float] = [0.1, 0.9],
    dc_filter_range: List[int] = [0, 12000],
    bimodal_fit: bool = False,
    use_skewed_super_lorentzian: bool = True,
) -> Tuple:
    """
    Fit Lorentzian (or skewed super-Lorentzian) peak to FFT signal.

    Parameters:
        config (dict): configuration dictionary
        paths (Paths): paths to data, figures, and fit files
        file_id (int): filename snippet
        fft (np.ndarray): FFT signal, shape (N, 2) [frequency Hz, amplitude]
        signal_proportion (float): portion of peak (by height) to include in fit
        frequency_bounds (List[float]): [min, max] frequency bounds for peak search [GHz]
        dc_filter_range (List[int]): [start, end] bin indices to zero (DC + tail rejection)
        bimodal_fit (bool): whether to perform bimodal peak fitting (Lorentzian form only)
        use_skewed_super_lorentzian (bool): use skewed super-Lorentzian for asymmetric peaks

    Returns:
        Tuple of 11:
            - saw_frequency [Hz]
            - saw_frequency_error [Hz]
            - fwhm [Hz]
            - tau [s]
            - snr [dB]
            - frequency_bounds [GHz]
            - fit_function (callable)
            - popt (np.ndarray)
            - fft_segment (np.ndarray): slice used for fitting
            - fft_full (np.ndarray): full spectrum (frequency in GHz, normalized amplitude)
            - lorentzian_curve (np.ndarray): evaluated fit curve on 500 pts over frequency_bounds
    """
    if bimodal_fit and use_skewed_super_lorentzian:
        raise ValueError(
            "bimodal_fit is not compatible with use_skewed_super_lorentzian; "
            "disable one of them"
        )

    start, end = dc_filter_range
    fft[:, 0] = fft[:, 0] / 1e9 
    fft[:start, 1] = 0
    if 0 < end < len(fft):
        fft[end:, 1] = 0

    min_freq, max_freq = frequency_bounds
    in_bounds = (fft[:, 0] >= min_freq) & (fft[:, 0] <= max_freq)
    if not in_bounds.any():
        raise ValueError(
            f"No FFT bins within frequency_bounds={frequency_bounds} GHz"
        )
    bound_indices = np.where(in_bounds)[0]
    search_start = max(start, int(bound_indices[0]))
    search_end = int(bound_indices[-1]) + 1
    if 0 < end < search_end:
        search_end = end
    if search_end <= search_start:
        raise ValueError(
            "dc_filter_range excludes the entire frequency_bounds window"
        )

    peak_idx = search_start + int(np.argmax(fft[search_start:search_end, 1]))
    peak_loc = float(fft[peak_idx, 0])

    max_value = float(fft[peak_idx, 1])
    if max_value <= 0:
        raise ValueError("FFT peak is non-positive; cannot normalize")
    fft[:, 1] /= max_value

    if signal_proportion != 1.0:
        threshold = signal_proportion * fft[peak_idx, 1]
        neg_idx = peak_idx
        while neg_idx > search_start and fft[neg_idx, 1] > threshold:
            neg_idx -= 1
        pos_idx = peak_idx
        while pos_idx < search_end - 1 and fft[pos_idx, 1] > threshold:
            pos_idx += 1
    else:
        neg_idx = search_start
        pos_idx = search_end

    if use_skewed_super_lorentzian:
        fit_function = skewed_super_lorentzian_function
        initial_guess = [1e-2, peak_loc, 0.05, 0, 0.5, -0.5]
        lower_bounds = [0, min_freq, 1e-3, 0, 0.1, -2.0]
        upper_bounds = [1, max_freq, 0.2, 1, 0.9, 2.0]
    else:
        fit_function = lorentzian_function
        initial_guess = [1e-4, peak_loc, 1e-2, 0]
        lower_bounds = [0, min_freq, 1e-3, 0]
        upper_bounds = [1, max_freq, 0.05, 1]

    bounds = (lower_bounds, upper_bounds)

    popt, pcov = curve_fit(
        fit_function, fft[neg_idx:pos_idx, 0], fft[neg_idx:pos_idx, 1],
        p0=initial_guess, bounds=bounds,
    )

    if use_skewed_super_lorentzian:
        _, x0, W, _, _, _ = popt
        _, x0_error, _, _, _, _ = np.sqrt(np.diag(pcov))
    else:
        _, x0, W, _ = popt
        _, x0_error, _, _ = np.sqrt(np.diag(pcov))

    saw_frequency = x0 * 1e9
    saw_frequency_error = x0_error * 1e9
    fwhm = 2 * W * 1e9
    tau = 1 / (np.pi * fwhm)

    popt2 = None
    if bimodal_fit:
        bimodal_start = max(search_start, int(round(0.1 * peak_idx)))
        bimodal_end = min(search_end, int(round(0.75 * peak_idx)))
        if bimodal_end - bimodal_start < 5:
            raise ValueError("bimodal_fit: search window has too few points")

        residual = fft[:, 1] - fit_function(fft[:, 0], *popt)
        peak_idx2 = bimodal_start + int(np.argmax(residual[bimodal_start:bimodal_end]))
        peak_loc2 = float(fft[peak_idx2, 0])

        initial_guess2 = [1e-4, peak_loc2, 0.01, 0]
        popt2, pcov2 = curve_fit(
            fit_function, fft[:, 0], residual, p0=initial_guess2, bounds=bounds,
        )
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

    def combined_fit(x):
        y = fit_function(x, *popt)
        if popt2 is not None:
            y = y + fit_function(x, *popt2)
        return y

    residual_full = fft[:, 1] - combined_fit(fft[:, 0])
    signal_power = float(np.mean(fft[:, 1] ** 2))
    noise_power = float(np.mean(residual_full ** 2))
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    if config['plot']['fft_lorentzian']:
        plot_fft_lorentzian(paths, file_id, fft[neg_idx:pos_idx], frequency_bounds, fit_function, popt)

    lorentzian_curve = combined_fit(np.linspace(min_freq, max_freq, 500))
    return (
        saw_frequency, saw_frequency_error, fwhm, tau, snr,
        frequency_bounds, fit_function, popt,
        fft[neg_idx:pos_idx], fft, lorentzian_curve,
    )
