from typing import Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from src.core.path import Paths
from src.analysis.signal_process import process_signal
from src.analysis.fft import fft
from src.analysis.lorentzian import lorentzian_fit
from src.analysis.functions import tgs_function
from src.core.plots import plot_tgs, plot_combined

def tgs_fit(config: dict, paths: Paths, file_idx: int, pos_file: str, neg_file: str, grating_spacing: float, signal_proportion: float= 0.9, maxfev: float= 1000000) -> Tuple[Union[float, np.ndarray]]:
    """
    Fit transient grating spectroscopy (TGS) response equation to experimentally collected signal.

    This function processes the input TGS signal, performs thermal and acoustic fits, and returns 
    the fitted parameters along with their standard errors (1σ).

    Parameters:
        config (dict): configuration dictionary
        paths (Paths): paths to data, figures, and fit files
        file_idx (int): index of the positive signal file
        pos_file (str): positive signal file path
        neg_file (str): negative signal file path
        grating_spacing (float): grating spacing of TGS probe [µm]
        signal_proportion (float): proportion of signal to use for fitting (0.0 to 1.0)
        maxfev (int): maximum number of function evaluations

    Returns:
        Tuple containing:
            start_time (float): start time of the fit [s]
            A (float): thermal signal amplitude [W/m²]
            A_err (float): thermal signal amplitude error [W/m²]
            B (float): acoustic signal amplitude [W/m²]
            B_err (float): acoustic signal amplitude error [W/m²]
            C (float): signal offset [W/m²]
            C_err (float): signal offset error [W/m²]
            alpha (float): thermal diffusivity [m²/s]
            alpha_err (float): thermal diffusivity error [m²/s]
            beta (float): displacement-reflectance ratio [s⁰⋅⁵]
            beta_err (float): displacement-reflectance ratio error [s⁰⋅⁵]
            theta (float): acoustic phase [rad]
            theta_err (float): acoustic phase error [rad]
            tau (float): acoustic decay time [s]
            tau_err (float): acoustic decay time error [s]
            f (float): surface acoustic wave frequency [Hz]
            f_err (float): surface acoustic wave frequency error [Hz]
            signal (np.ndarray): full processed signal [N, [time, amplitude]]
            fitted_signal (np.ndarray): truncated signal used for fitting [M, [time, amplitude]]

    Notes:
        The fitting process includes:
            1. Initial thermal fit
            2. FFT analysis and Lorentzian fit for acoustic parameters
            3. Iterative beta fitting
            4. Functional fit including thermal and acoustic components
    """
    #  Isolate the name for each signal for labelling
    ID1 = str(pos_file).rsplit("/")
    ID2 = ID1[-1].rsplit("\\") #deals with both filepath conventions
    file_id = ID2[-1].split(".txt")[0]
    
    # Process signal and build fit functions
    signal, max_time, start_time, start_idx = process_signal(config, paths, file_idx, pos_file, neg_file, grating_spacing, **config['signal_process'])
    end_idx = int(len(signal) * signal_proportion) + start_idx
    functional_function, thermal_function = tgs_function(start_time, grating_spacing)

    # Thermal fit
    thermal_p0 = [0.05, 5e-4]
    popt, _ = curve_fit(lambda x, A, alpha: thermal_function(x, A, 0, 0, alpha, 0, 0, 0, 0), signal[:, 0], signal[:, 1], p0=thermal_p0)
    A, alpha = popt
    if alpha <= 0:
        alpha = 1e-6
    
    # Lorentzian fit on FFT of SAW signal
    saw_signal = np.column_stack([
        signal[:, 0], 
        signal[:, 1] - thermal_function(signal[:, 0], A, 0, 0, alpha, 0, 0, 0, 0)
    ])
    fft_signal = fft(saw_signal, **config['fft'])
    f, _, _, tau, _, frequency_bounds, lorentzian_function, lorentzian_popt = lorentzian_fit(config, paths, file_idx, fft_signal, **config['lorentzian'])

    # Iteratively fit beta (displacement-reflectance ratio)
    q = 2 * np.pi / (grating_spacing * 1e-6)
    for _ in range(10):
        displacement = q * np.sqrt(alpha / np.pi)
        reflectance = (q ** 2 * alpha + 1 / (2 * max_time))
        beta = displacement / reflectance
        popt, _ = curve_fit(lambda x, A, alpha: thermal_function(x, A, 0, 0, alpha, beta, 0, 0, 0), signal[start_idx:end_idx, 0], signal[start_idx:end_idx, 1], p0=thermal_p0)
        A, alpha = popt

    # Functional fit
    functional_p0 = [0.05, 0.05, 0, alpha, beta, 0, tau, f]
    tgs_popt, tgs_pcov = curve_fit(functional_function, signal[start_idx:end_idx, 0], signal[start_idx:end_idx, 1], p0=functional_p0, maxfev=maxfev)
    A, B, C, alpha, beta, theta, tau, f = tgs_popt
    A_err, B_err, C_err, alpha_err, beta_err, theta_err, tau_err, f_err = np.sqrt(np.diag(tgs_pcov))

    if config['plot']['tgs']:
        plot_tgs(paths, file_id, signal, start_idx, functional_function, thermal_function, tgs_popt, config['plot']['settings']['num_points'])

    if config['plot']['signal_process'] and config['plot']['fft_lorentzian'] and config['plot']['tgs']:
        plot_combined(paths, file_id, signal, max_time, start_time, start_idx, functional_function, thermal_function, tgs_popt,
                     fft_signal, frequency_bounds, lorentzian_function, lorentzian_popt, config['plot']['settings']['num_points'])

    return start_idx, start_time, grating_spacing, A, A_err, B, B_err, C, C_err, alpha, alpha_err, beta, beta_err, theta, theta_err, tau, tau_err, f, f_err, signal
