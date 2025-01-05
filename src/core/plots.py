import matplotlib
import numpy as np

from matplotlib import pyplot as plt

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

def plot_tgs(paths, file_idx, signal, start_idx, functional_function, thermal_function, fit_params, num_points=None):
    if num_points is None:
        num_points = len(signal)
    x_raw, y_raw = signal[:num_points, 0], signal[:num_points, 1]
    x_fit = signal[start_idx:num_points, 0]

    plt.figure(figsize=(10, 6))
    plt.plot(x_raw * 1e9, y_raw * 1e3, linestyle='-', color='black', linewidth=2, label='Signal')
    plt.plot(x_fit * 1e9, functional_function(x_fit, *fit_params) * 1e3, linestyle='-', color='blue', linewidth=2, label='Functional Fit')
    plt.plot(x_fit * 1e9, thermal_function(x_fit, *fit_params) * 1e3, linestyle='-', color='red', linewidth=2, label='Thermal Fit')

    plt.xlabel('Time [ns]', fontsize=16, labelpad=10)
    plt.ylabel('Signal Amplitude [mV]', fontsize=16, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.75)
    plt.legend(fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    save_dir = paths.figure_dir / 'tgs'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'tgs-{file_idx:04d}.png'
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_fft_lorentzian(paths, file_idx, fft, frequency_bounds, lorentzian_function, popt):
    frequencies, amplitudes = fft[:, 0], fft[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, amplitudes, linestyle='-', color='black', linewidth=2, label='FFT Signal')
    
    x_smooth = np.linspace(min(frequencies), max(frequencies), 1000)
    y_fit = lorentzian_function(x_smooth, *popt)
    plt.plot(x_smooth, y_fit, linestyle='--', color='red', linewidth=2, label='Lorentzian Fit')

    y_range = plt.ylim()
    plt.vlines(frequency_bounds[0], y_range[0], y_range[1], color='purple', linestyle='--', linewidth=2, label='Frequency Bounds')
    plt.vlines(frequency_bounds[1], y_range[0], y_range[1], color='purple', linestyle='--', linewidth=2)
    plt.xlim(0, 1)
    
    plt.xlabel('Frequency [GHz]', fontsize=16, labelpad=10)
    plt.ylabel('Intensity [A.U.]', fontsize=16, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.75)
    plt.legend(fontsize=16)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    save_dir = paths.figure_dir / 'fft-lorentzian'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'fft-lorentzian-{file_idx:04d}.png'
    plt.savefig(save_path, dpi=600)
    plt.close()
    
def plot_signal_process(paths, file_idx, signal, max_time, start_time, num_points=None):
    if num_points is None:
        num_points = len(signal)
    time, amplitude = signal[:num_points, 0], signal[:num_points, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time * 1e9, amplitude * 1e3, linestyle='-', color='black', linewidth=2, label='Signal')
    
    y_range = plt.ylim()
    plt.vlines(max_time * 1e9, y_range[0], y_range[1], color='blue', linestyle='--', linewidth=2, label='Max Time')
    plt.vlines(start_time * 1e9, y_range[0], y_range[1], color='red', linestyle='--', linewidth=2, label='Start Time')
    
    plt.xlabel('Time [ns]', fontsize=16, labelpad=10)
    plt.ylabel('Signal Amplitude [mV]', fontsize=16, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.75)
    plt.legend(fontsize=16)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    save_dir = paths.figure_dir / 'signal-processed'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'signal-processed-{file_idx:04d}.png'
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_combined(paths, file_idx, signal, max_time, start_time, start_idx, functional_function, thermal_function, tgs_popt,
                 fft, frequency_bounds, lorentzian_function, lorentzian_popt, num_points=None):
    if num_points is None:
        num_points = len(signal)
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1])
    
    ax1 = fig.add_subplot(gs[:, 0])
    x_raw, y_raw = signal[:num_points, 0], signal[:num_points, 1]
    x_fit = signal[start_idx:num_points, 0]
    ax1.plot(x_raw * 1e9, y_raw * 1e3, '-k', linewidth=2, label='Signal')
    ax1.plot(x_fit * 1e9, functional_function(x_fit, *tgs_popt) * 1e3, '-b', linewidth=2, label='Functional Fit')
    ax1.plot(x_fit * 1e9, thermal_function(x_fit, *tgs_popt) * 1e3, '-r', linewidth=2, label='Thermal Fit')
    ax1.set_xlabel('Time [ns]', fontsize=14, labelpad=10)
    ax1.set_ylabel('Signal Amplitude [mV]', fontsize=14, labelpad=10)
    ax1.tick_params(labelsize=10)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.75)
    ax1.legend(fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = fig.add_subplot(gs[0, 1])
    frequencies, amplitudes = fft[:, 0], fft[:, 1]
    ax2.plot(frequencies, amplitudes, '-k', linewidth=2, label='FFT Signal')
    x_smooth = np.linspace(min(frequencies), max(frequencies), 1000)
    y_fit = lorentzian_function(x_smooth, *lorentzian_popt)
    ax2.plot(x_smooth, y_fit, '--r', linewidth=2, label='Lorentzian Fit')
    y_range = ax2.get_ylim()
    ax2.vlines(frequency_bounds[0], y_range[0], y_range[1], color='purple', linestyle='--', linewidth=2, label='Frequency Bounds')
    ax2.vlines(frequency_bounds[1], y_range[0], y_range[1], color='purple', linestyle='--', linewidth=2)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Frequency [GHz]', fontsize=14, labelpad=10)
    ax2.set_ylabel('Intensity [A.U.]', fontsize=14, labelpad=10)
    ax2.tick_params(labelsize=10)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.75)
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax3 = fig.add_subplot(gs[1, 1])
    time, amplitude = signal[:num_points, 0], signal[:num_points, 1]
    ax3.plot(time * 1e9, amplitude * 1e3, '-k', linewidth=2, label='Signal')
    y_range = ax3.get_ylim()
    ax3.vlines(max_time * 1e9, y_range[0], y_range[1], color='blue', linestyle='--', linewidth=2, label='Max Time')
    ax3.vlines(start_time * 1e9, y_range[0], y_range[1], color='red', linestyle='--', linewidth=2, label='Start Time')
    ax3.set_xlabel('Time [ns]', fontsize=14, labelpad=10)
    ax3.set_ylabel('Signal Amplitude [mV]', fontsize=14, labelpad=10)
    ax3.tick_params(labelsize=10)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.75)
    ax3.legend(fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    save_dir = paths.figure_dir / 'combined'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'combined-{file_idx:04d}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
