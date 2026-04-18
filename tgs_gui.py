#!/usr/bin/env python3
"""
PyTGS Graphical Interface - MATLAB-style layout with full configuration editing
"""

import sys
import os
import threading
import queue
import logging
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Import PyTGS modules
from src.analysis.signal_process import process_signal
from src.analysis.fft import fft
from src.analysis.lorentzian import lorentzian_fit
from src.core.path import Paths

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

# Redirect logging to a queue for GUI display
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
    def emit(self, record):
        # Only show INFO and above in GUI (no DEBUG)
        if record.levelno >= logging.INFO:
            self.queue.put(self.format(record))


class ToolTip:
    """Create tooltips for widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.enter)
        widget.bind('<Leave>', self.leave)
    
    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", foreground="#000000",
                        relief=tk.SOLID, borderwidth=1,
                        font=("Arial", 9))
        label.pack()
    
    def leave(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class TGSApp:
    VERSION = "0.1.4-alpha"

    def __init__(self, root):
        self.root = root
        self.root.title("PyTGS v0.1.3-alpha - Transient Grating Spectroscopy Analyzer")
        self.root.geometry("1200x900")  # Increased size for better visibility
        
        # Set theme to match system
        self.setup_theme()
        
        # Load configuration
        self.config = self.load_config()
        self.calibrated_spacing = None
        
        # File storage
        self.pos_files = []
        self.neg_files = []
        self.calib_pos_file = ''
        self.calib_neg_file = ''
        self.baseline_pos_file = ''
        self.baseline_neg_file = ''
        self.file_to_fit_plot_data = {}  # Mapping from file_id to fit plot data

        # Store fit results for each processed file
        self.fit_results = []  # List of dicts containing file info and plot path
        self.file_to_plot_path = {}  # Mapping from file_id to combined plot path
        self.file_to_fit_params = {}  # Mapping from file_id to fit parameters
        self.current_results_log_path = None  # Store the current results log path
        
        # Setup logging
        self.setup_logging()
        
        # Build GUI
        self.create_widgets()
        self.root.bind('<Configure>', self.on_window_resize)
        self.resize_timer = None

        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.running_job = None

        # Show placeholder text in results table (now after treeview is created)
        self.clear_results_table()
        
        # Time tracking for batch processing
        self.batch_start_time = None
        self.first_run_duration = None
        self.current_run_start_time = None
    
    def setup_theme(self):
        """Configure modern dark theme for the application"""
        style = ttk.Style()
        
        # Define modern dark colors
        self.bg_color = '#1e1e1e'      # Main window background (dark)
        self.panel_bg = '#2d2d2d'       # Panel background (slightly lighter)
        self.fg_color = '#e0e0e0'       # Text color
        self.select_color = '#404040'
        self.button_color = '#3c3c3c'
        self.entry_bg = '#3c3c3c'
        self.trough_color = '#2d2d2d'
        self.accent_color = '#0078d4'
        self.accent_hover = '#005a9e'
        
        # Configure root window background
        self.root.configure(background=self.bg_color)
        
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        
        # A style for frames inside panels
        style.configure('Panel.TFrame',
            background=self.panel_bg
        )

        # Configure ttk styles for modern dark theme - remove all borders
        style.configure('.',
            background=self.bg_color,
            foreground=self.fg_color,
            fieldbackground=self.entry_bg,
            troughcolor=self.trough_color,
            selectbackground=self.select_color,
            selectforeground=self.fg_color,
            borderwidth=0,
            focuscolor='',
            font=('Arial', 9)
        )
        
        # Configure PanedWindow to have no border
        style.configure('TPanedWindow',
            background=self.bg_color,
            borderwidth=0,
            sashrelief='flat',
            sashthickness=4
        )
        style.configure('TPanedWindow.Sash',
            background=self.bg_color,
            borderwidth=0
        )
        
        # Configure TLabelframe (panels) with modern styling - no borders
        style.configure('TLabelframe',
            background=self.panel_bg,
            foreground=self.fg_color,
            darkcolor=self.panel_bg,
            bordercolor=self.panel_bg,
            borderwidth=0,
            relief='flat'
        )
        
        # Configure TLabelframe labels - use panel background
        style.configure('TLabelframe.Label',
            background=self.panel_bg,  # Use panel background instead of empty
            foreground=self.fg_color,
            font=('Arial', 10, 'bold')
        )
        
        # Configure Panel.TFrame for frames inside LabelFrames
        style.configure('Panel.TFrame',
            background=self.panel_bg
        )

        style.configure('Panel.TCombobox',
            fieldbackground=self.panel_bg,
            background=self.panel_bg,
            foreground=self.fg_color,
            selectbackground=self.select_color,
            selectforeground=self.fg_color,
            borderwidth=1,
            padding=6,
            arrowsize=12,
            font=('Arial', 9)
        )
        
        # Configure TFrame - use bg_color for main frames, panel_bg for child frames
        style.configure('TFrame',
            background=self.bg_color
        )
        
        # Configure TLabel - use panel_bg for labels in panels
        style.configure('TLabel',
            background=self.panel_bg,  # Change from '' to self.panel_bg
            foreground=self.fg_color,
            font=('Arial', 9)
        )
        
        # Configure TButton with modern styling
        style.configure('TButton',
            background=self.button_color,
            foreground=self.fg_color,
            borderwidth=0,
            focusthickness=0,
            padding=8,
            font=('Arial', 9)
        )
        
        style.map('TButton',
            background=[('active', self.select_color), ('pressed', self.accent_color)],
            foreground=[('active', self.fg_color)]
        )
        
        # Configure Accent button
        style.configure('Accent.TButton',
            background=self.accent_color,
            foreground='white',
            borderwidth=0,
            padding=8,
            font=('Arial', 9, 'bold')
        )
        style.map('Accent.TButton',
            background=[('active', self.accent_hover), ('pressed', self.accent_hover)]
        )
        
        # Configure TEntry
        style.configure('TEntry',
            fieldbackground=self.entry_bg,
            foreground=self.fg_color,
            insertcolor=self.fg_color,
            borderwidth=1,
            padding=6,
            font=('Arial', 9)
        )

        style.configure('Panel.TEntry',
            fieldbackground=self.panel_bg,
            background=self.panel_bg,  # Add this line
            foreground=self.fg_color,
            insertcolor=self.fg_color,
            borderwidth=1,
            padding=6,
            font=('Arial', 9)
        )
        
        # Configure TSpinbox
        style.configure('TSpinbox',
            fieldbackground=self.entry_bg,
            foreground=self.fg_color,
            insertcolor=self.fg_color,
            borderwidth=1,
            padding=6,
            font=('Arial', 9)
        )
        
        # Configure TCombobox with modern styling
        style.configure('TCombobox',
            fieldbackground=self.entry_bg,
            background=self.entry_bg,
            foreground=self.fg_color,
            selectbackground=self.select_color,
            selectforeground=self.fg_color,
            borderwidth=1,
            padding=6,
            arrowsize=12,
            font=('Arial', 9)
        )
        
        # Style for Combobox dropdown list
        style.configure('TCombobox.listbox',
            background=self.entry_bg,
            foreground=self.fg_color,
            selectbackground=self.select_color,
            selectforeground=self.fg_color,
            borderwidth=0,
            font=('Arial', 9)
        )
        
        # Map combobox colors
        style.map('TCombobox',
            fieldbackground=[('readonly', self.entry_bg)],
            background=[('readonly', self.entry_bg)],
            foreground=[('readonly', self.fg_color)]
        )
        
        # Configure TCheckbutton with modern styling - use panel_bg for background
        style.configure('TCheckbutton',
            background=self.panel_bg,  # Change from self.bg_color to self.panel_bg
            foreground=self.fg_color,
            indicatorsize=16,
            font=('Arial', 9)
        )
        style.map('TCheckbutton',
            background=[('active', self.panel_bg), ('pressed', self.panel_bg)],
            foreground=[('active', self.fg_color)]
        )
        
        # Configure TProgressbar - white
        style.configure('TProgressbar',
            background='#ffffff',
            troughcolor=self.trough_color,
            borderwidth=0
        )
        
        # Configure TScrollbar
        style.configure('Vertical.TScrollbar',
            background=self.button_color,
            troughcolor=self.trough_color,
            borderwidth=0,
            arrowcolor=self.fg_color
        )
        
        style.configure('Horizontal.TScrollbar',
            background=self.button_color,
            troughcolor=self.trough_color,
            borderwidth=0,
            arrowcolor=self.fg_color
        )
        
        # Configure Listbox (tk widget)
        self.root.option_add('*Listbox*background', self.entry_bg)
        self.root.option_add('*Listbox*foreground', self.fg_color)
        self.root.option_add('*Listbox*selectBackground', self.select_color)
        self.root.option_add('*Listbox*selectForeground', self.fg_color)
        self.root.option_add('*Listbox*borderWidth', 0)
        self.root.option_add('*Listbox*relief', 'flat')
        self.root.option_add('*Listbox*font', ('Arial', 9))
    
    def on_window_resize(self, event):
        """Maintain sash position proportion when window is resized (debounced)"""
        # Debounce: wait for resize to finish before updating
        if self.resize_timer:
            self.root.after_cancel(self.resize_timer)
        
        # Schedule the actual resize handling after a short delay
        self.resize_timer = self.root.after(100, lambda: self._handle_resize(event))

    def _handle_resize(self, event):
        """Actually handle the resize after debouncing"""
        # Only update sash positions
        if hasattr(self, 'paned_window') and event.widget == self.root:
            total_width = self.root.winfo_width()
            if total_width > 100:
                new_sash_pos = int(total_width * 0.42)  # Match the initial proportion
                try:
                    self.paned_window.sashpos(0, new_sash_pos)
                except:
                    pass
        
        if hasattr(self, 'right_paned') and event.widget == self.root:
            total_height = self.root.winfo_height()
            if total_height > 100:
                new_sash_pos = int(total_height * 0.4)
                try:
                    self.right_paned.sashpos(0, new_sash_pos)
                except:
                    pass

    def set_initial_sash(self):
        """Set initial divider position"""
        try:
            total_width = self.root.winfo_width()
            if total_width > 100:
                # Set left panel to 42% of width (1.2x of 35% = 42%)
                self.paned_window.sashpos(0, int(total_width * 0.42))
            
            if hasattr(self, 'right_paned'):
                total_height = self.root.winfo_height()
                if total_height > 100:
                    self.right_paned.sashpos(0, int(total_height * 0.35))
        except:
            pass

    def load_config(self):
        """Load config.yaml from the current directory or create default."""
        # Check if running as executable
        if getattr(sys, 'frozen', False):
            # Running as compiled executable - look for config in the same directory as the exe
            config_path = Path(os.path.dirname(sys.executable)) / "config.yaml"
        else:
            # Running as script
            config_path = Path("config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            default = {
                "path": "example",
                "study_names": None,
                "idxs": None,
                "signal_process": {
                    "heterodyne": "di-homodyne",
                    "null_point": 2,
                    "initial_samples": 50,
                    "baseline_correction": {"enabled": False, "pos": None, "neg": None}
                },
                "fft": {"signal_proportion": 1.0, "use_derivative": True, "analysis_type": "psd"},
                "lorentzian": {
                    "signal_proportion": 1,
                    "frequency_bounds": [0.1, 0.9],
                    "dc_filter_range": [0, 50000],
                    "bimodal_fit": False,
                    "use_skewed_super_lorentzian": False
                },
                "tgs": {"grating_spacing": 3.5276, "signal_proportion": 1, "maxfev": 1000000},
                "plot": {
                    "signal_process": True,
                    "fft_lorentzian": True,
                    "tgs": True,
                    "settings": {"num_points": None}
                }
            }
            return default
    
    def save_config(self):
        """Save current config to config.yaml."""
        # Save to the appropriate location
        if getattr(sys, 'frozen', False):
            config_path = Path(os.path.dirname(sys.executable)) / "config.yaml"
        else:
            config_path = Path("config.yaml")
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)
        self.log_message("Configuration saved to config.yaml")
    
    def setup_logging(self):
        """Redirect logging to a queue for display in GUI."""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Queue handler for GUI (INFO and above only)
        self.queue_handler = QueueHandler(log_queue)
        self.queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.queue_handler)
        
        # Suppress matplotlib debug messages
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        self.poll_log_queue()
    
    def poll_log_queue(self):
        """Periodically update the log text widget."""
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + '\n')
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.poll_log_queue)
    
    def log_message(self, msg, level=logging.INFO):
        """Helper to log a message."""
        self.logger.log(level, msg)
    
    def create_widgets(self):
        """Create main layout - MATLAB style with adjustable divider"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create a PanedWindow for adjustable left/right panels
        self.paned_window = ttk.PanedWindow(main_frame, orient='horizontal')
        self.paned_window.pack(fill='both', expand=True)

        # Left panel for controls - increased weight to 3 for wider panel
        left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(left_panel, weight=3)  # Changed from 2 to 3

        # Right panel for log output
        right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(right_panel, weight=2)  # Changed from 1 to 2
        
        # Set initial sash positions after window is drawn
        self.root.after(100, self.set_initial_sash)
        
        # Build left panel sections
        self.build_calibration_section(left_panel)
        self.build_parameters_section(left_panel)
        self.build_batch_section(left_panel)
        
        # Build right panel (log)
        self.build_log_section(right_panel)
    
    def add_tooltip(self, widget, text):
        """Add a tooltip to a widget"""
        return ToolTip(widget, text)

    def build_calibration_section(self, parent):
        """Section 1: Calibration (grating spacing)"""
        frame = ttk.LabelFrame(parent, text="1. Calibration (grating spacing)", padding=(15, 10))
        frame.pack(fill='x', pady=(0, 15))
        
        # File selection row
        file_row = ttk.Frame(frame, style='Panel.TFrame')
        file_row.pack(fill='x', pady=8)
        
        btn = ttk.Button(file_row, text="Select calibration files", 
                  command=self.select_calib_files)
        btn.pack(side='left', padx=(0, 15))
        self.add_tooltip(btn, "Select POS and NEG files for calibration (must be a matching pair)")
        
        # Make label background transparent
        self.calib_file_label = ttk.Label(file_row, text="No files selected", foreground=self.fg_color)
        self.calib_file_label.pack(side='left', fill='x', expand=True)
        
        # Grating spacing row
        spacing_row = ttk.Frame(frame, style='Panel.TFrame')
        spacing_row.pack(fill='x', pady=8)
        
        lbl = ttk.Label(spacing_row, text="Grating (µm):", background='')
        lbl.pack(side='left', padx=(0, 15))
        self.add_tooltip(lbl, "Calculated grating spacing from calibration (or manually entered)")
        
        self.grating_edit = ttk.Entry(spacing_row, width=15)
        self.grating_edit.pack(side='left', padx=(0, 15))
        self.grating_edit.insert(0, "0")
        self.add_tooltip(self.grating_edit, "Grating spacing in micrometers (µm)")
        
        btn = ttk.Button(spacing_row, text="Run calibration", command=self.run_calibration)
        btn.pack(side='left')
        self.add_tooltip(btn, "Run calibration using selected files and known sound speed (2665.9 m/s)")
        
        self.sound_speed = 2665.9
    
    def build_parameters_section(self, parent):
        """Section 2: Global parameters with modern styling"""
        frame = ttk.LabelFrame(parent, text="2. Global parameters", padding=(15, 10))
        frame.pack(fill='x', pady=(0, 15))
        
        # Start point and checkboxes row
        start_row = ttk.Frame(frame, style='Panel.TFrame')
        start_row.pack(fill='x', pady=8)
        
        lbl = ttk.Label(start_row, text="Start point (1-4):", background='')
        lbl.pack(side='left', padx=(0, 15))
        self.add_tooltip(lbl, "Null point selection for TGS signal phase analysis (valid range: 1-4)")
        
        self.start_point_var = tk.IntVar(value=self.config['signal_process']['null_point'])
        spin = ttk.Spinbox(start_row, from_=1, to=4, textvariable=self.start_point_var, width=5)
        spin.pack(side='left', padx=(0, 25))
        self.add_tooltip(spin, "Select which null point to use for fitting (1-4)")
        
        self.two_saw_var = tk.BooleanVar(value=self.config['lorentzian']['bimodal_fit'])
        cb = ttk.Checkbutton(start_row, text="Two SAW", variable=self.two_saw_var)
        cb.pack(side='left', padx=(0, 25))
        self.add_tooltip(cb, "Enable bimodal Lorentzian fitting for two SAW peaks")
        
        self.close_plots_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(start_row, text="Disable plot saving", variable=self.close_plots_var)
        cb.pack(side='left')
        self.add_tooltip(cb, "Check to disable saving plot images (uncheck to save plots)")
        
        # Baseline row
        baseline_row = ttk.Frame(frame, style='Panel.TFrame')
        baseline_row.pack(fill='x', pady=8)
        
        self.baseline_var = tk.BooleanVar(value=self.config['signal_process']['baseline_correction']['enabled'])
        cb = ttk.Checkbutton(baseline_row, text="Use baseline", variable=self.baseline_var,
                       command=self.toggle_baseline_ui)
        cb.pack(side='left', padx=(0, 15))
        self.add_tooltip(cb, "Enable baseline correction using reference files")
        
        self.baseline_button = ttk.Button(baseline_row, text="Select baseline", 
                                         command=self.select_baseline_files, state='disabled')
        self.baseline_button.pack(side='left', padx=(0, 15))
        self.add_tooltip(self.baseline_button, "Select POS and NEG baseline reference files")
        
        self.baseline_file_label = ttk.Label(baseline_row, text="No baseline selected", foreground=self.fg_color, background='')
        self.baseline_file_label.pack(side='left', fill='x', expand=True)
        
        btn = ttk.Button(frame, text="Edit All Parameters...", command=self.open_config_editor)
        btn.pack(fill='x', pady=12)
        self.add_tooltip(btn, "Open full configuration editor with all advanced settings")
    
    def build_batch_section(self, parent):
        """Section 3: Batch processing queue"""
        frame = ttk.LabelFrame(parent, text="3. Batch processing queue", padding=(15, 10))
        frame.pack(fill='both', expand=True)
        
        # Button row - use Panel.TFrame style
        btn_row = ttk.Frame(frame, style='Panel.TFrame')
        btn_row.pack(fill='x', pady=(0, 12))
        
        # Make buttons smaller and more compact
        btn = ttk.Button(btn_row, text="Add files", command=self.add_batch_files, width=12)
        btn.pack(side='left', padx=(0, 10))
        self.add_tooltip(btn, "Add TGS files to batch processing queue (select both POS and NEG files)")
        
        btn = ttk.Button(btn_row, text="Clear queue", command=self.clear_queue, width=12)
        btn.pack(side='left', padx=(0, 10))
        self.add_tooltip(btn, "Clear all files from the batch queue")
        
        self.remove_button = ttk.Button(btn_row, text="Remove", command=self.remove_selected_item, width=12)
        self.remove_button.pack(side='left', padx=(0, 10))
        self.add_tooltip(self.remove_button, "Remove selected run from queue and results log")
        
        self.stop_button = ttk.Button(btn_row, text="Stop", command=self.stop_batch_processing, state='disabled', width=12)
        self.stop_button.pack(side='left')
        self.add_tooltip(self.stop_button, "Stop the current batch processing (finishes current file)")
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill='both', expand=True, pady=(0, 12))
        
        scrollbar = ttk.Scrollbar(list_frame)
        
        self.batch_listbox = tk.Listbox(
            list_frame, 
            yscrollcommand=scrollbar.set,
            bg='#3c3c3c',
            fg='#f0f0f0',
            selectbackground='#404040',
            selectforeground='#f0f0f0',
            relief='flat',
            borderwidth=0,
            highlightthickness=0,
            font=('Consolas', 9)
        )
        self.batch_listbox.pack(side='left', fill='both', expand=True)
        
        def update_scrollbar_visibility(event=None):
            if self.batch_listbox.size() > 0:
                if self.batch_listbox.bbox(0) is not None:
                    last_index = self.batch_listbox.size() - 1
                    last_bbox = self.batch_listbox.bbox(last_index)
                    if last_bbox:
                        listbox_height = self.batch_listbox.winfo_height()
                        if last_bbox[1] + last_bbox[3] > listbox_height:
                            scrollbar.pack(side='right', fill='y')
                        else:
                            scrollbar.pack_forget()
                    else:
                        scrollbar.pack_forget()
                else:
                    scrollbar.pack_forget()
            else:
                scrollbar.pack_forget()
        
        self.batch_listbox.bind('<<ListboxSelect>>', update_scrollbar_visibility)
        self.batch_listbox.bind('<Configure>', update_scrollbar_visibility)
        
        def on_items_changed():
            self.root.after(100, update_scrollbar_visibility)
        
        original_insert = self.batch_listbox.insert
        original_delete = self.batch_listbox.delete
        
        def custom_insert(*args):
            original_insert(*args)
            on_items_changed()
        
        def custom_delete(*args):
            original_delete(*args)
            on_items_changed()
        
        self.batch_listbox.insert = custom_insert
        self.batch_listbox.delete = custom_delete
        
        scrollbar.config(command=self.batch_listbox.yview)
        self.batch_listbox.bind('<<ListboxSelect>>', self.on_batch_item_selected)
        
        # Export settings
        export_frame = ttk.LabelFrame(frame, text="Export Settings", padding=(12, 8))
        export_frame.pack(fill='x', pady=(0, 10))

        # Use Panel.TFrame style for the inner frame to match panel background
        log_row = ttk.Frame(export_frame, style='Panel.TFrame')
        log_row.pack(fill='x', pady=8)

        lbl = ttk.Label(log_row, text="Output Folder:")
        lbl.pack(side='left', padx=(0, 15))
        self.add_tooltip(lbl, "Path where the results log file will be saved")

        self.log_file_var = tk.StringVar()
        entry = ttk.Entry(log_row, textvariable=self.log_file_var, style='Panel.TEntry')
        entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.add_tooltip(entry, "File path for saving results (space-delimited format)")

        btn = ttk.Button(log_row, text="Browse...", command=self.browse_log_file)
        btn.pack(side='left')
        self.add_tooltip(btn, "Browse to select log file location")
        
        # Run button - normal button
        self.run_button = ttk.Button(export_frame, text="Run batch process", command=self.run_batch)
        self.run_button.pack(fill='x', pady=12)
        self.add_tooltip(self.run_button, "Start batch processing of all files in the queue")
        
        # Progress bar - start with determinate mode
        self.progress = ttk.Progressbar(export_frame, mode='determinate', maximum=100, value=0)
        self.progress.pack(fill='x', pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        lbl = ttk.Label(export_frame, textvariable=self.status_var, background='')
        lbl.pack(pady=5)
        self.add_tooltip(lbl, "Current processing status")
    
    def build_log_section(self, parent):
        """Right panel - plot preview on top, results table in middle, log output below"""
        self.right_paned = ttk.PanedWindow(parent, orient='vertical')
        self.right_paned.pack(fill='both', expand=True)
        
        # Top frame for plot preview - use a custom frame with title bar
        plot_container = ttk.Frame(self.right_paned)
        self.right_paned.add(plot_container, weight=2)
        
        # Create title bar for the plot container
        title_bar = ttk.Frame(plot_container)
        title_bar.pack(fill='x', pady=(0, 5))
        
        # Create title label with window background color (darker grey)
        title_label = ttk.Label(title_bar, text="Fit Preview", font=('Arial', 10, 'bold'),
                                background=self.bg_color, foreground=self.fg_color)
        title_label.pack(side='left', padx=(12, 0))
        
        # Add save button to the title bar
        save_btn = ttk.Button(title_bar, text="Save Plot", command=self.save_current_plot, width=10)
        save_btn.pack(side='right', padx=(0, 12))
        self.add_tooltip(save_btn, "Save the current plot as PNG, PDF, or SVG")
        
        # Create a frame for the plot content - no border, just panel background
        plot_frame = ttk.Frame(plot_container)
        plot_frame.pack(fill='both', expand=True, padx=12, pady=(0, 8))
        
        # Create matplotlib figure for interactive plotting
        self.preview_fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.panel_bg, tight_layout=True)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_ax.set_facecolor(self.panel_bg)
        self.preview_ax.tick_params(colors=self.fg_color)
        self.preview_ax.xaxis.label.set_color(self.fg_color)
        self.preview_ax.yaxis.label.set_color(self.fg_color)
        self.preview_ax.title.set_color(self.fg_color)
        for spine in self.preview_ax.spines.values():
            spine.set_color(self.fg_color)

        # Create inset axes for FFT plot (initially hidden)
        self.inset_ax = self.preview_ax.inset_axes([0.65, 0.6, 0.3, 0.35])
        self.inset_ax.set_facecolor(self.panel_bg)
        self.inset_ax.tick_params(colors=self.fg_color)
        self.inset_ax.xaxis.label.set_color(self.fg_color)
        self.inset_ax.yaxis.label.set_color(self.fg_color)
        for spine in self.inset_ax.spines.values():
            spine.set_color(self.fg_color)
        self.inset_ax.set_visible(False)

        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=plot_frame)
        self.preview_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        # Store the latest fit data for redrawing
        self.current_fit_data = None
        
        # MIDDLE FRAME: Results table
        results_frame = ttk.LabelFrame(self.right_paned, text="Fit Parameters", padding=(12, 8))
        self.right_paned.add(results_frame, weight=1)

        # Create Treeview for results table
        self.results_tree = ttk.Treeview(results_frame, columns=('value',), show='tree headings', height=8)
        self.results_tree.heading('#0', text='Parameter')
        self.results_tree.heading('value', text='Value')

        # Configure column widths
        self.results_tree.column('#0', width=200, minwidth=150)
        self.results_tree.column('value', width=200, minwidth=150)

        # Pack the treeview without scrollbar
        self.results_tree.pack(fill='both', expand=True)
        
        # Configure treeview colors for dark theme
        style = ttk.Style()
        style.configure("Treeview", background=self.entry_bg, foreground=self.fg_color, fieldbackground=self.entry_bg)
        style.configure("Treeview.Heading", background=self.button_color, foreground=self.fg_color)
        style.map('Treeview', background=[('selected', self.select_color)])
        
        # Bottom frame for log output
        log_frame = ttk.LabelFrame(self.right_paned, text="Log Output", padding=(12, 8))
        self.right_paned.add(log_frame, weight=2)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD, 
            height=20,
            bg='#252526',
            fg='#d4d4d4',
            insertbackground='#f0f0f0',
            selectbackground='#264f78',
            selectforeground='#f0f0f0',
            relief='flat',
            borderwidth=0,
            font=('Consolas', 9)
        )
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        btn_frame = ttk.Frame(log_frame)
        btn_frame.pack(fill='x', pady=(8, 0))
        btn = ttk.Button(btn_frame, text="Clear Log", command=self.clear_log)
        btn.pack(side='right')
        self.add_tooltip(btn, "Clear the log output display")

    def save_current_plot(self):
        """Save the currently displayed plot to a file"""
        if not hasattr(self, 'current_fit_data') or self.current_fit_data is None:
            # Try to get the currently selected fit data
            selection = self.batch_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "No plot to save. Please select a processed run first.")
                return
            
            idx = selection[0]
            if idx >= len(self.pos_files):
                messagebox.showwarning("Warning", "No plot available for the selected item.")
                return
            
            pos_file = self.pos_files[idx]
            file_id = Path(pos_file).stem
            
            if file_id in self.file_to_fit_plot_data:
                fit_data = self.file_to_fit_plot_data[file_id]
            else:
                messagebox.showwarning("Warning", "No plot data available for the selected run.")
                return
        else:
            fit_data = self.current_fit_data
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG image", "*.svg"),
                ("All files", "*.*")
            ],
            title="Save Plot As"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Create a new figure for saving with higher DPI
            save_fig = Figure(figsize=(8, 5), dpi=300, facecolor=self.panel_bg, tight_layout=True)
            save_ax = save_fig.add_subplot(111)
            save_ax.set_facecolor(self.panel_bg)
            
            # Copy the current plot to the new figure
            font_props = {'family': 'Arial', 'color': self.fg_color}
            
            # Plot raw data
            if 'time_raw' in fit_data and 'signal_raw' in fit_data:
                save_ax.plot(fit_data['time_raw'] * 1e9, fit_data['signal_raw'] * 1e3, 
                            '-', color='white', linewidth=0.75, alpha=0.7, label='Raw Data', zorder=1)
            
            # Plot functional fit
            if 'time_fit' in fit_data and 'fit_signal' in fit_data:
                save_ax.plot(fit_data['time_fit'] * 1e9, fit_data['fit_signal'] * 1e3, 
                            '-', color='#4dabf7', linewidth=1, label='Functional Fit', alpha=0.9, zorder=2)
            
            # Plot thermal fit
            if 'time_fit' in fit_data and 'thermal_signal' in fit_data:
                save_ax.plot(fit_data['time_fit'] * 1e9, fit_data['thermal_signal'] * 1e3, 
                            '-', color='#ff6b6b', linewidth=1, label='Thermal Fit', alpha=0.9, zorder=3)
            
            # Set labels and title
            save_ax.set_xlabel('Time [ns]', fontdict=font_props, fontsize=11)
            save_ax.set_ylabel('Signal Amplitude [mV]', fontdict=font_props, fontsize=11)
            if 'title' in fit_data:
                save_ax.set_title(fit_data['title'], fontdict=font_props, fontsize=12, pad=10)
            
            save_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=self.fg_color)
            
            # Legend
            legend = save_ax.legend(loc='lower right', fontsize=9, 
                                facecolor=self.panel_bg, edgecolor=self.fg_color)
            legend.get_frame().set_alpha(0.8)
            for text in legend.get_texts():
                text.set_color('white')
                text.set_fontfamily('Arial')
            
            # Add inset FFT if available
            has_fft = ('fft_freq' in fit_data and fit_data['fft_freq'] is not None and 
                    len(fit_data['fft_freq']) > 0 and 'fft_amp' in fit_data and fit_data['fft_amp'] is not None)
            
            if has_fft:
                # Create inset axes
                inset_ax = save_ax.inset_axes([0.6, 0.55, 0.35, 0.4])
                inset_ax.set_facecolor(self.panel_bg)
                
                # Configure inset spines
                for spine in inset_ax.spines.values():
                    spine.set_color(self.fg_color)
                    spine.set_linewidth(1)
                
                # Set tick colors
                inset_ax.tick_params(colors=self.fg_color, labelsize=9)
                
                # Plot FFT data
                fft_freq = fit_data['fft_freq']
                fft_amp = fit_data['fft_amp']
                
                valid_mask = np.isfinite(fft_freq) & np.isfinite(fft_amp)
                if np.any(valid_mask):
                    fft_freq_valid = fft_freq[valid_mask]
                    fft_amp_valid = fft_amp[valid_mask]
                    
                    inset_ax.plot(fft_freq_valid, fft_amp_valid, 
                                '-', color='white', linewidth=0.5, alpha=0.7, label='FFT')
                    
                    # Plot Lorentzian fit if available
                    if ('lorentzian_fit' in fit_data and fit_data['lorentzian_fit'] is not None and
                        'lorentzian_freq' in fit_data and fit_data['lorentzian_freq'] is not None):
                        lorentz_freq = fit_data['lorentzian_freq']
                        lorentz_fit = fit_data['lorentzian_fit']
                        
                        valid_lorentz = np.isfinite(lorentz_freq) & np.isfinite(lorentz_fit)
                        if np.any(valid_lorentz):
                            inset_ax.plot(lorentz_freq[valid_lorentz], lorentz_fit[valid_lorentz], 
                                        '-', color='#ff6b6b', linewidth=0.75, alpha=0.9, label='Lorentzian')
                    
                    # Set axis limits around the peak
                    peak_idx = np.argmax(fft_amp_valid)
                    peak_freq = fft_freq_valid[peak_idx]
                    freq_min = max(0.05, peak_freq - 0.2)
                    freq_max = min(1.0, peak_freq + 0.2)
                    inset_ax.set_xlim(freq_min, freq_max)
                    
                    inset_ax.set_xlabel('Frequency [GHz]', fontsize=9, fontfamily='Arial', color=self.fg_color)
                    inset_ax.set_ylabel('Intensity', fontsize=9, fontfamily='Arial', color=self.fg_color)
                    inset_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=self.fg_color)
                    
                    inset_legend = inset_ax.legend(loc='upper right', fontsize=7,
                                                facecolor=self.panel_bg, edgecolor=self.fg_color)
                    inset_legend.get_frame().set_alpha(0.8)
                    for text in inset_legend.get_texts():
                        text.set_color('white')
                        text.set_fontfamily('Arial')
            
            # Save the figure
            save_fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=self.panel_bg)
            plt.close(save_fig)
            
            self.log_message(f"Plot saved to: {file_path}")
            messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}")
            
        except Exception as e:
            self.log_message(f"Error saving plot: {str(e)}", logging.ERROR)
            messagebox.showerror("Error", f"Failed to save plot:\n{str(e)}")
    
    def create_interactive_plot(self, fit_data):
        """Create an interactive plot with main TGS curve and inset FFT"""
        
        # Store current fit data for saving
        self.current_fit_data = fit_data

        # Clear the main axes
        self.preview_ax.clear()
        self.preview_ax.set_facecolor(self.panel_bg)
        
        # Set font properties for all text - Arial throughout
        font_props = {'family': 'Arial', 'color': self.fg_color}
        
        # Set tick params
        self.preview_ax.tick_params(colors=self.fg_color, labelsize=9)
        for label in self.preview_ax.get_xticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(9)
            label.set_color(self.fg_color)
        for label in self.preview_ax.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(9)
            label.set_color(self.fg_color)
        
        # Plot raw data
        if 'time_raw' in fit_data and 'signal_raw' in fit_data:
            self.preview_ax.plot(fit_data['time_raw'] * 1e9, fit_data['signal_raw'] * 1e3, 
                                '-', color='white', linewidth=0.75, alpha=0.7, label='Raw Data', zorder=1)
        
        # Plot functional fit
        if 'time_fit' in fit_data and 'fit_signal' in fit_data:
            self.preview_ax.plot(fit_data['time_fit'] * 1e9, fit_data['fit_signal'] * 1e3, 
                                '-', color='#4dabf7', linewidth=1, label='Functional Fit', alpha=0.9, zorder=2)
        
        # Plot thermal fit
        if 'time_fit' in fit_data and 'thermal_signal' in fit_data:
            self.preview_ax.plot(fit_data['time_fit'] * 1e9, fit_data['thermal_signal'] * 1e3, 
                                '-', color='#ff6b6b', linewidth=1, label='Thermal Fit', alpha=0.9, zorder=3)
        
        # Set labels and title with Arial font
        self.preview_ax.set_xlabel('Time [ns]', fontdict=font_props, fontsize=11)
        self.preview_ax.set_ylabel('Signal Amplitude [mV]', fontdict=font_props, fontsize=11)
        if 'title' in fit_data:
            self.preview_ax.set_title(fit_data['title'], fontdict=font_props, fontsize=12, pad=10)
        
        self.preview_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=self.fg_color)
        
        # Legend moved to bottom right with scaled size (0.7x)
        legend = self.preview_ax.legend(loc='lower right', fontsize=9, 
                                    facecolor=self.panel_bg, edgecolor=self.fg_color)
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color('white')
            text.set_fontfamily('Arial')
            text.set_fontsize(int(9 * 0.7))  # Scale legend text by 0.7
        
        # Handle inset FFT plot - Remove and recreate inset each time to ensure it works
        # First, remove any existing inset
        if hasattr(self, 'inset_ax'):
            try:
                self.inset_ax.remove()
            except:
                pass
        
        # Check if we have FFT data to display
        has_fft = ('fft_freq' in fit_data and fit_data['fft_freq'] is not None and 
                len(fit_data['fft_freq']) > 0 and 'fft_amp' in fit_data and fit_data['fft_amp'] is not None)
        
        if has_fft:
            # Create a new inset axes - 1.1x larger than before (was [0.6, 0.55, 0.35, 0.4], now scaled by 1.1)
            # New size: width 0.35*1.1=0.385, height 0.4*1.1=0.44
            self.inset_ax = self.preview_ax.inset_axes([0.59, 0.53, 0.385, 0.44])
            self.inset_ax.set_facecolor(self.panel_bg)
            
            # Configure inset spines
            for spine in self.inset_ax.spines.values():
                spine.set_color(self.fg_color)
                spine.set_linewidth(1)
            
            # Set tick colors and Arial font
            self.inset_ax.tick_params(colors=self.fg_color, labelsize=9)
            for label in self.inset_ax.get_xticklabels():
                label.set_fontfamily('Arial')
                label.set_fontsize(9)
                label.set_color(self.fg_color)
            for label in self.inset_ax.get_yticklabels():
                label.set_fontfamily('Arial')
                label.set_fontsize(9)
                label.set_color(self.fg_color)
            
            # Plot FFT data
            fft_freq = fit_data['fft_freq']
            fft_amp = fit_data['fft_amp']

            # Filter valid data
            valid_mask = np.isfinite(fft_freq) & np.isfinite(fft_amp)
            if np.any(valid_mask):
                fft_freq_valid = fft_freq[valid_mask]
                fft_amp_valid = fft_amp[valid_mask]
                
                # Plot FFT
                self.inset_ax.plot(fft_freq_valid, fft_amp_valid, 
                                '-', color='white', linewidth=0.5, alpha=0.7, label='FFT')
                
                # Plot Lorentzian fit if available (line thickness reduced by half: was 1.5, now 0.75)
                if ('lorentzian_fit' in fit_data and fit_data['lorentzian_fit'] is not None and
                    'lorentzian_freq' in fit_data and fit_data['lorentzian_freq'] is not None):
                    lorentz_freq = fit_data['lorentzian_freq']
                    lorentz_fit = fit_data['lorentzian_fit']
                    
                    # Ensure we have valid data
                    valid_lorentz = np.isfinite(lorentz_freq) & np.isfinite(lorentz_fit)
                    if np.any(valid_lorentz):
                        lorentz_freq_valid = lorentz_freq[valid_lorentz]
                        lorentz_fit_valid = lorentz_fit[valid_lorentz]
                        
                        # Plot Lorentzian fit with reduced line thickness (0.75 instead of 1.5)
                        self.inset_ax.plot(lorentz_freq_valid, lorentz_fit_valid, 
                                        '-', color='#ff6b6b', linewidth=0.75, alpha=0.9, label='Lorentzian Fit')
                
                # Set axis limits around the peak
                peak_idx = np.argmax(fft_amp_valid)
                peak_freq = fft_freq_valid[peak_idx]
                freq_min = max(0.05, peak_freq - 0.2)
                freq_max = min(1.0, peak_freq + 0.2)
                self.inset_ax.set_xlim(freq_min, freq_max)
                
                # Set labels with Arial font
                self.inset_ax.set_xlabel('Frequency [GHz]', fontsize=9, fontfamily='Arial', color=self.fg_color)
                self.inset_ax.set_ylabel('Intensity', fontsize=9, fontfamily='Arial', color=self.fg_color)
                self.inset_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=self.fg_color)
                
                # Inset legend with Arial font (scaled by 0.7: was 7, now 5)
                inset_legend = self.inset_ax.legend(loc='upper right', fontsize=5,
                                                facecolor=self.panel_bg, edgecolor=self.fg_color)
                inset_legend.get_frame().set_alpha(0.8)
                for text in inset_legend.get_texts():
                    text.set_color('white')
                    text.set_fontfamily('Arial')
                    text.set_fontsize(5)
                
                self.inset_ax.set_zorder(10)
        
        # Adjust layout and redraw
        self.preview_fig.tight_layout()
        self.preview_canvas.draw_idle()

    def format_value_with_error(self, value, error, conversion_factor=1.0):
        """Format a value with its error using ± symbol, handle inf/NaN"""
        # Check if error is inf or NaN
        if error is None or np.isinf(error) or np.isnan(error):
            return "BF"
        
        # Check if value is inf or NaN
        if value is None or np.isinf(value) or np.isnan(value):
            return "BF"
        
        try:
            # Convert to float and apply conversion factor
            val = float(value) * conversion_factor
            err = float(error) * conversion_factor
            
            if err == 0:
                return f"{val:.6e}"
            
            # Determine appropriate significant figures
            if abs(val) < 0.001 or abs(val) > 1000:
                # Use scientific notation
                err_str = f"{err:.2e}"
                if 'e' in err_str:
                    exp = int(err_str.split('e')[-1])
                    val_scaled = val / (10**exp)
                    err_scaled = err / (10**exp)
                    if err_scaled < 10:
                        decimals = max(2, 2 - int(np.floor(np.log10(abs(err_scaled)))) + 1)
                    else:
                        decimals = 0
                    return f"{val_scaled:.{decimals}f} ± {err_scaled:.{decimals}f}e{exp}"
            else:
                # For numbers in reasonable range
                if err < 0.0001:
                    decimals = 6
                elif err < 0.001:
                    decimals = 5
                elif err < 0.01:
                    decimals = 4
                elif err < 0.1:
                    decimals = 3
                elif err < 1:
                    decimals = 2
                else:
                    decimals = 1
                
                return f"{val:.{decimals}f} ± {err:.{decimals}f}"
        except:
            return "BF"
    
    def update_results_table(self, fit_params):
        """Update the results table with fit parameters using Treeview"""
        # Check if results_tree exists
        if not hasattr(self, 'results_tree'):
            return
        
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Define parameters to display with their units and conversion factors
        parameters = [
            ('A', 'A', 'W·m⁻²', 1.0),
            ('B', 'B', 'W·m⁻²', 1.0),
            ('C', 'C', 'W·m⁻²', 1.0),
            ('alpha', 'α (thermal diffusivity)', 'mm²·s⁻¹', 1000000.0),
            ('beta', 'β', 's⁰·⁵', 1.0),
            ('theta', 'θ', 'rad', 1.0),
            ('tau', 'τ', 'ns', 1000000000.0),
            ('f', 'f', 'MHz', 1e-6),
        ]
        
        # Add parameters to the tree
        for key, display_name, unit, conv_factor in parameters:
            if key in fit_params and fit_params[key] is not None:
                value = fit_params[key]
                error_key = f"{key}_err"
                error = fit_params.get(error_key, None)
                formatted_value = self.format_value_with_error(value, error, conv_factor)
                
                # Format the display string
                if unit:
                    display_text = f"{display_name} ({unit})"
                else:
                    display_text = display_name
                
                # Insert into tree
                self.results_tree.insert('', 'end', text=display_text, values=(formatted_value,))
            else:
                display_text = f"{display_name} ({unit})" if unit else display_name
                self.results_tree.insert('', 'end', text=display_text, values=('Not available',))
    
    def store_fit_parameters(self, file_id, params):
        """Store fit parameters for a file"""
        self.file_to_fit_params[file_id] = params
    
    def update_preview_from_combined_plot(self, file_id, data_dir):
        """Update the preview with the latest combined plot from the fit"""
        combined_plot_path = data_dir / 'figures' / 'combined' / f'combined-{file_id}.png'
        
        if combined_plot_path.exists():
            self.current_preview_path = combined_plot_path
            
            # Store mapping from file_id to plot path for later retrieval
            self.file_to_plot_path[file_id] = str(combined_plot_path)
            
            try:
                # Use PIL to load a downsampled version
                from PIL import Image
                
                img = Image.open(str(combined_plot_path))
                
                # Get widget size for display
                widget = self.preview_canvas.get_tk_widget()
                target_width = widget.winfo_width()
                target_height = widget.winfo_height()
                
                # Downsample if image is larger than display area
                if target_width > 0 and target_height > 0:
                    img_width, img_height = img.size
                    if img_width > target_width * 1.5 or img_height > target_height * 1.5:
                        scale = min(target_width / img_width, target_height / img_height) * 1.2
                        new_size = (int(img_width * scale), int(img_height * scale))
                        if new_size[0] > 0 and new_size[1] > 0:
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert PIL image to numpy array for matplotlib
                import numpy as np
                img_array = np.array(img)
                
                # Clear the axes completely
                self.preview_ax.clear()
                self.preview_ax.set_facecolor(self.bg_color)
                
                # Display image
                self.preview_ax.imshow(img_array, aspect='auto')
                self.preview_ax.axis('off')
                self.preview_ax.set_title(f"Latest Fit: {file_id}", color=self.fg_color, fontsize=10, fontfamily='Arial')
                
                # Use blitting for faster updates
                self.preview_canvas.draw_idle()
                
                # Clean up
                del img
                del img_array
                
            except Exception as e:
                self.log_message(f"Could not load preview image: {str(e)}", logging.DEBUG)
                self._show_placeholder_text(f"Preview unavailable\n{file_id}")
        else:
            self._show_placeholder_text(f"Waiting for fit results\n{file_id}")

    def on_batch_item_selected(self, event):
        """Handle selection of a batch list item - display its fitted plot and parameters"""
        selection = self.batch_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        if idx >= len(self.pos_files):
            return
        
        # Get the file identifier
        pos_file = self.pos_files[idx]
        file_id = Path(pos_file).stem
        
        self.log_message(f"Selected: {file_id}", logging.DEBUG)
        self.log_message(f"Has plot data: {hasattr(self, 'file_to_fit_plot_data') and file_id in self.file_to_fit_plot_data}", logging.DEBUG)
        self.log_message(f"Has fit params: {file_id in self.file_to_fit_params}", logging.DEBUG)
        
        # First try to use stored plot data
        if hasattr(self, 'file_to_fit_plot_data') and file_id in self.file_to_fit_plot_data:
            fit_data = self.file_to_fit_plot_data[file_id]
            self.create_interactive_plot(fit_data)
        elif file_id in self.file_to_fit_params:
            # Try to reconstruct from parameters
            self.display_selected_plot(None, file_id, idx)
        else:
            self._show_placeholder_text(f"No fit data available\n{file_id}")
        
        # Display the parameters if available (ONCE)
        if file_id in self.file_to_fit_params:
            self.update_results_table(self.file_to_fit_params[file_id])
        else:
            self.clear_results_table()
    
    def display_selected_plot(self, plot_path, file_id, idx):
        """Display selected fit data as an interactive plot with inset FFT (fallback)"""
        try:
            # First check if we have stored plot data
            if hasattr(self, 'file_to_fit_plot_data') and file_id in self.file_to_fit_plot_data:
                fit_data = self.file_to_fit_plot_data[file_id]
                self.create_interactive_plot(fit_data)
                return
            
            # If not, try to get from file_id in fit_params
            if file_id in self.file_to_fit_params:
                fit_params = self.file_to_fit_params[file_id]
                
                # We need to reconstruct the fit data. This requires loading the original signal
                # and the fit results. Since the fit parameters are stored, we can regenerate the fits.
                pos_file = self.pos_files[idx]
                data_dir = Path(pos_file).parent
                
                # Try to load the signal data from the fit directory
                signal_path = data_dir / 'fit' / 'signal.json'
                fit_csv_path = data_dir / 'fit' / 'fit.csv'
                
                fit_data = {'title': file_id}
                
                # Load the signal data
                if signal_path.exists():
                    import json
                    with open(signal_path, 'r') as f:
                        signals = json.load(f)
                    # Find the signal for this file (by index)
                    if idx < len(signals):
                        signal_array = np.array(signals[idx])
                        fit_data['time_raw'] = signal_array[:, 0]
                        fit_data['signal_raw'] = signal_array[:, 1]
                
                # Load the fit parameters to regenerate the fits
                if fit_csv_path.exists():
                    import pandas as pd
                    df = pd.read_csv(fit_csv_path)
                    # Find the row for this file
                    row = df[df['run_name'].str.contains(file_id)]
                    if not row.empty:
                        # Extract parameters
                        A = row['A[Wm^-2]'].values[0] if 'A[Wm^-2]' in row else fit_params.get('A', 0)
                        B = row['B[Wm^-2]'].values[0] if 'B[Wm^-2]' in row else fit_params.get('B', 0)
                        C = row['C[Wm^-2]'].values[0] if 'C[Wm^-2]' in row else fit_params.get('C', 0)
                        alpha = row['alpha[m^2s^-1]'].values[0] if 'alpha[m^2s^-1]' in row else fit_params.get('alpha', 1e-6)
                        beta = row['beta[s^0.5]'].values[0] if 'beta[s^0.5]' in row else fit_params.get('beta', 0)
                        theta = row['theta[rad]'].values[0] if 'theta[rad]' in row else fit_params.get('theta', 0)
                        tau = row['tau[s]'].values[0] if 'tau[s]' in row else fit_params.get('tau', 1e-6)
                        f = row['f[Hz]'].values[0] if 'f[Hz]' in row else fit_params.get('f', 1e6)
                        start_time = row['start_time'].values[0] if 'start_time' in row else 0
                        grating_spacing = row['grating_spacing[µm]'].values[0] if 'grating_spacing[µm]' in row else 6.4
                        
                        # Generate time points for fit
                        if 'time_raw' in fit_data:
                            time_raw = fit_data['time_raw']
                            # Use the same time range as raw data but starting from start_time
                            time_fit = np.linspace(max(start_time, time_raw[0]), time_raw[-1], 1000)
                            
                            # Create fit functions
                            from src.analysis.functions import tgs_function
                            functional_func, thermal_func = tgs_function(start_time, grating_spacing)
                            
                            # Generate fits
                            fit_data['time_fit'] = time_fit
                            fit_data['fit_signal'] = functional_func(time_fit, A, B, C, alpha, beta, theta, tau, f)
                            fit_data['thermal_signal'] = thermal_func(time_fit, A, B, C, alpha, beta, theta, tau, f)
                
                # Try to load FFT data
                fft_data_path = data_dir / 'figures' / 'fft-lorentzian' / f'fft-lorentzian-{file_id}.png'
                # For FFT data, we need to reconstruct from the fit parameters
                if 'f' in fit_params and fit_params['f'] is not None:
                    # Generate a synthetic FFT based on the Lorentzian fit
                    freq_range = np.linspace(0.05, 0.95, 500)
                    from src.analysis.functions import lorentzian_function
                    f_ghz = fit_params['f'] / 1e9 if fit_params['f'] is not None else 0.5
                    # Estimate width from tau
                    width = 1 / (2 * np.pi * fit_params.get('tau', 1e-6) * 1e9) if fit_params.get('tau') else 0.05
                    fit_data['fft_freq'] = freq_range
                    fit_data['fft_amp'] = lorentzian_function(freq_range, 1.0, f_ghz, width, 0)
                    fit_data['lorentzian_freq'] = freq_range
                    fit_data['lorentzian_fit'] = lorentzian_function(freq_range, 1.0, f_ghz, width, 0)
                
                # Create the interactive plot
                self.create_interactive_plot(fit_data)
                pass
            else:
                self._show_placeholder_text(f"No fit data available\n{file_id}")
                    
        except Exception as e:
            self.log_message(f"Could not create interactive plot: {str(e)}", logging.DEBUG)
            self._show_placeholder_text(f"Error creating plot\n{file_id}")

    def clear_results_table(self):
        """Clear the results table and show placeholder message"""
        # Check if results_tree exists before trying to use it
        if hasattr(self, 'results_tree'):
            # Clear existing items
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Insert placeholder message
            self.results_tree.insert('', 'end', text='Click a processed run to view fit parameters', values=('',))
        
        # Check if preview_ax exists before trying to clear the plot
        if hasattr(self, 'preview_ax'):
            # Clear the plot and show placeholder
            self.preview_ax.clear()
            self.preview_ax.set_facecolor(self.panel_bg)
            self.preview_ax.text(0.5, 0.5, 'Click a processed run\nto view fit results',
                                ha='center', va='center', transform=self.preview_ax.transAxes,
                                color=self.fg_color, fontsize=12, fontfamily='Arial')
            self.preview_ax.set_xlim(0, 1)
            self.preview_ax.set_ylim(0, 1)
            self.preview_ax.set_xticks([])
            self.preview_ax.set_yticks([])
            
            if hasattr(self, 'inset_ax'):
                self.inset_ax.set_visible(False)
            
            if hasattr(self, 'preview_canvas'):
                self.preview_canvas.draw()

    def remove_selected_item(self):
        """Remove the selected item from the batch queue and from the results log"""
        selection = self.batch_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an item to remove.")
            return
        
        idx = selection[0]
        
        # Confirm with user
        if not messagebox.askyesno("Confirm Remove", 
                                   f"Remove '{self.batch_listbox.get(idx)}' from queue and results log?"):
            return
        
        # Get the file ID before removal
        if idx < len(self.pos_files):
            pos_file = self.pos_files[idx]
            file_id = Path(pos_file).stem
            
            # Remove from file lists
            self.pos_files.pop(idx)
            self.neg_files.pop(idx)
            
            # Remove from plot mapping if exists
            if file_id in self.file_to_plot_path:
                del self.file_to_plot_path[file_id]
            
            # Remove from fit parameters mapping if exists
            if file_id in self.file_to_fit_params:
                del self.file_to_fit_params[file_id]
            
            # Remove from listbox
            self.batch_listbox.delete(idx)
            
            # Update numbering in listbox (renumber remaining items)
            for i in range(idx, self.batch_listbox.size()):
                current_text = self.batch_listbox.get(i)
                # Extract the part after the number
                text_parts = current_text.split('] ', 1)
                if len(text_parts) == 2:
                    new_text = f"[{i+1}] {text_parts[1]}"
                    self.batch_listbox.delete(i)
                    self.batch_listbox.insert(i, new_text)
            
            # Remove corresponding entry from results log if it exists
            self.remove_from_results_log(file_id)
            
            self.log_message(f"Removed '{file_id}' from queue and results log")
            
            # Clear preview if this was the selected item
            if self.batch_listbox.size() == 0:
                self._show_placeholder_text("Queue empty\nAdd files to process")
                self.clear_results_table()
            elif idx == 0 and self.batch_listbox.size() > 0:
                # Select the first item if it exists
                self.batch_listbox.selection_set(0)
                self.on_batch_item_selected(None)
        else:
            self.batch_listbox.delete(idx)
            self.log_message("Removed item from queue")
    
    def remove_from_results_log(self, file_id):
        """Remove a row corresponding to file_id from the results log file"""
        # Check if we have a current results log path
        log_path = self.log_file_var.get().strip()
        if not log_path:
            # Try to determine if there's a default log file
            if self.pos_files:
                first_file = Path(self.pos_files[0])
                default_log = first_file.parent / f"{first_file.stem.split('-POS')[0]}_postprocessing.txt"
                if default_log.exists():
                    log_path = str(default_log)
                else:
                    return
            else:
                return
        
        log_file = Path(log_path)
        if not log_file.exists():
            return
        
        try:
            # Read all lines from the log file
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) <= 1:
                return  # Only header or empty
            
            # Header is first line
            header = lines[0]
            data_lines = lines[1:]
            
            # Find and remove lines that start with the file_id (run_name)
            new_data_lines = []
            removed = False
            
            for line in data_lines:
                # Check if the first word (run_name) matches file_id
                parts = line.strip().split()
                if parts and parts[0] == file_id:
                    removed = True
                    self.log_message(f"Removed '{file_id}' from results log")
                else:
                    new_data_lines.append(line)
            
            # Write back the file without the removed line
            if removed:
                with open(log_file, 'w') as f:
                    f.write(header)
                    f.writelines(new_data_lines)
                self.log_message(f"Updated results log: {log_file}")
            else:
                self.log_message(f"File '{file_id}' not found in results log", logging.DEBUG)
                
        except Exception as e:
            self.log_message(f"Error updating results log: {str(e)}", logging.WARNING)

    def cleanup_old_plots(self, data_dir, max_to_keep=50):
        """Delete old combined plot images to save disk space (optional)"""
        try:
            combined_dir = data_dir / 'figures' / 'combined'
            if combined_dir.exists():
                # Get all combined plot files sorted by modification time
                plot_files = sorted(combined_dir.glob('combined-*.png'), key=lambda x: x.stat().st_mtime)
                
                # Delete older files beyond max_to_keep
                for old_file in plot_files[:-max_to_keep]:
                    try:
                        old_file.unlink()
                    except:
                        pass
        except:
            pass  # Silent fail for cleanup

    def _show_placeholder_text(self, text):
        """Show placeholder text when no image is available"""
        self.preview_ax.clear()
        self.preview_ax.set_facecolor(self.panel_bg)
        self.preview_ax.text(0.5, 0.5, text,
                            ha='center', va='center', transform=self.preview_ax.transAxes,
                            color=self.fg_color, fontsize=12, fontfamily='Arial', wrap=True)
        self.preview_ax.set_xlim(0, 1)
        self.preview_ax.set_ylim(0, 1)
        self.preview_ax.set_xticks([])
        self.preview_ax.set_yticks([])
        self.preview_canvas.draw()

    def open_config_editor(self):
        """Open a scrollable window to edit all config parameters with intuitive controls"""
        editor_window = tk.Toplevel(self.root)
        editor_window.title("Configuration Editor - All Parameters")
        
        # Let the window size to its content, but set a reasonable default
        editor_window.geometry("")  # Clear any forced geometry
        editor_window.update_idletasks()  # Update to get natural size
        # Set minimum size to prevent it from being too small
        editor_window.minsize(950, 610)
        
        # Apply dark theme to editor window
        editor_window.configure(background=self.bg_color)
        
        # Make it modal
        editor_window.transient(self.root)
        editor_window.grab_set()
        
        # Create main frame
        main_frame = ttk.Frame(editor_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create a canvas with scrollbar for the entire content
        canvas = tk.Canvas(main_frame, background=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_reqwidth())
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Update canvas width when frame size changes
        def update_canvas_width(event):
            canvas.itemconfig(1, width=event.width)
        
        canvas.bind('<Configure>', update_canvas_width)
        
        # Store widgets for saving
        self.config_widgets = {}
        
        # Create a container frame for the two columns
        columns_frame = ttk.Frame(scrollable_frame)
        columns_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create left and right column frames with equal width distribution
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Define intuitive configuration sections
        sections = [
            {
                'column': left_column,
                'title': "Path Settings",
                'params': [
                    {
                        'key': 'path',
                        'label': 'Data Directory',
                        'tooltip': 'Directory containing input files',
                        'type': 'text',
                        'width': 45,
                        'default': ''
                    },
                    {
                        'key': 'study_names',
                        'label': 'Study Names',
                        'tooltip': 'Study names to fit (comma-separated, leave empty for all)',
                        'type': 'text',
                        'width': 45,
                        'default': ''
                    },
                    {
                        'key': 'idxs',
                        'label': 'File Indices',
                        'tooltip': 'Indices of files to fit (comma-separated, leave empty for all)',
                        'type': 'text',
                        'width': 45,
                        'default': ''
                    }
                ]
            },
            {
                'column': left_column,
                'title': "Signal Processing",
                'params': [
                    {
                        'key': 'signal_process.heterodyne',
                        'label': 'Detection Method',
                        'tooltip': 'Heterodyne detection method for TGS signal',
                        'type': 'select',
                        'options': ['di-homodyne', 'mono-homodyne'],
                        'value_type': str,
                        'default': 'di-homodyne'
                    },
                    {
                        'key': 'signal_process.null_point',
                        'label': 'Null Point',
                        'tooltip': 'Null point selection for TGS signal phase analysis (1-4)',
                        'type': 'select',
                        'options': ['1', '2', '3', '4'],
                        'value_type': int,
                        'default': '2'
                    },
                    {
                        'key': 'signal_process.initial_samples',
                        'label': 'Initial Samples',
                        'tooltip': 'Number of samples for initial correction and prominence calculation',
                        'type': 'number',
                        'value_type': int,
                        'default': 50
                    }
                ]
            },
            {
                'column': left_column,
                'title': "Baseline Correction",
                'params': [
                    {
                        'key': 'signal_process.baseline_correction.enabled',
                        'label': 'Enable Baseline',
                        'tooltip': 'Enable/disable baseline correction using reference files',
                        'type': 'toggle',
                        'on_text': 'Enabled',
                        'default': False
                    },
                    {
                        'key': 'signal_process.baseline_correction.pos',
                        'label': 'POS Reference',
                        'tooltip': 'Filename for positive reference baseline',
                        'type': 'text',
                        'width': 35,
                        'default': ''
                    },
                    {
                        'key': 'signal_process.baseline_correction.neg',
                        'label': 'NEG Reference',
                        'tooltip': 'Filename for negative reference baseline',
                        'type': 'text',
                        'width': 35,
                        'default': ''
                    }
                ]
            },
            {
                'column': left_column,
                'title': "FFT Analysis",
                'params': [
                    {
                        'key': 'fft.signal_proportion',
                        'label': 'Signal Proportion',
                        'tooltip': 'Proportion of signal to analyze (0.0 to 1.0)',
                        'type': 'number',
                        'value_type': float,
                        'default': 1.0
                    },
                    {
                        'key': 'fft.use_derivative',
                        'label': 'Use Derivative',
                        'tooltip': 'Use signal derivative instead of raw signal',
                        'type': 'toggle',
                        'on_text': 'Enabled',
                        'default': True
                    },
                    {
                        'key': 'fft.analysis_type',
                        'label': 'Analysis Type',
                        'tooltip': 'Analysis method for frequency domain',
                        'type': 'select',
                        'options': ['psd', 'fft'],
                        'value_type': str,
                        'default': 'psd'
                    }
                ]
            },
            {
                'column': right_column,
                'title': "Lorentzian Fitting",
                'params': [
                    {
                        'key': 'lorentzian.signal_proportion',
                        'label': 'Signal Proportion',
                        'tooltip': 'Proportion of signal to use for fitting (0.0 to 1.0)',
                        'type': 'number',
                        'value_type': float,
                        'default': 1.0
                    },
                    {
                        'key': 'lorentzian.frequency_bounds',
                        'label': 'Frequency Range (GHz)',
                        'tooltip': 'Frequency range for fitting in GHz [min, max]',
                        'type': 'range',
                        'value_type': float,
                        'default': [0.1, 0.9]
                    },
                    {
                        'key': 'lorentzian.dc_filter_range',
                        'label': 'DC Filter Range (Hz)',
                        'tooltip': 'DC filtering range in Hz [min, max]',
                        'type': 'range',
                        'value_type': int,
                        'default': [0, 50000]
                    },
                    {
                        'key': 'lorentzian.bimodal_fit',
                        'label': 'Two SAW Fit',
                        'tooltip': 'Enable bimodal Lorentzian fitting for two SAW peaks',
                        'type': 'toggle',
                        'on_text': 'Enabled',
                        'default': False
                    },
                    {
                        'key': 'lorentzian.use_skewed_super_lorentzian',
                        'label': 'Skewed Super-Lorentzian',
                        'tooltip': 'Use skewed super-Lorentzian for asymmetric peaks',
                        'type': 'toggle',
                        'on_text': 'Enabled',
                        'default': False
                    }
                ]
            },
            {
                'column': right_column,
                'title': "TGS Fitting",
                'params': [
                    {
                        'key': 'tgs.grating_spacing',
                        'label': 'Grating Spacing (µm)',
                        'tooltip': 'TGS probe grating spacing in micrometers',
                        'type': 'number',
                        'value_type': float,
                        'default': 3.5276
                    },
                    {
                        'key': 'tgs.signal_proportion',
                        'label': 'Signal Proportion',
                        'tooltip': 'Proportion of signal to use for fitting (0.0 to 1.0)',
                        'type': 'number',
                        'value_type': float,
                        'default': 1.0
                    },
                    {
                        'key': 'tgs.maxfev',
                        'label': 'Max Iterations',
                        'tooltip': 'Maximum number of iterations for final functional fit',
                        'type': 'number',
                        'value_type': int,
                        'default': 1000000
                    }
                ]
            },
            {
                'column': right_column,
                'title': "Plot Settings",
                'params': [
                    {
                        'key': 'plot.signal_process',
                        'label': 'Plot Processed Signal',
                        'tooltip': 'Enable/disable processed signal visualization',
                        'type': 'toggle',
                        'on_text': 'Show',
                        'default': True
                    },
                    {
                        'key': 'plot.fft_lorentzian',
                        'label': 'Plot FFT & Lorentzian',
                        'tooltip': 'Enable/disable FFT and Lorentzian fit visualization',
                        'type': 'toggle',
                        'on_text': 'Show',
                        'default': True
                    },
                    {
                        'key': 'plot.tgs',
                        'label': 'Plot TGS Fit',
                        'tooltip': 'Enable/disable TGS fit visualization',
                        'type': 'toggle',
                        'on_text': 'Show',
                        'default': True
                    },
                    {
                        'key': 'plot.settings.num_points',
                        'label': 'Plot Points',
                        'tooltip': 'Number of points to plot (leave empty for all)',
                        'type': 'number',
                        'value_type': int,
                        'default': None
                    }
                ]
            }
        ]
        
        # Build all sections
        for section in sections:
            self._build_intuitive_config_section(section['column'], section['title'], section['params'])
        
        # Buttons at the bottom
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill='x', pady=10)
        
        btn_save = ttk.Button(btn_frame, text="Save", command=lambda: self.save_intuitive_config(editor_window))
        btn_save.pack(side='right', padx=5)
        self.add_tooltip(btn_save, "Save all changes and close editor")
        
        btn_cancel = ttk.Button(btn_frame, text="Cancel", command=editor_window.destroy)
        btn_cancel.pack(side='right', padx=5)
        self.add_tooltip(btn_cancel, "Discard changes and close editor")
        
        btn_reload = ttk.Button(btn_frame, text="Reload from File", command=lambda: self.reload_config_to_editor(editor_window))
        btn_reload.pack(side='left')
        self.add_tooltip(btn_reload, "Reload configuration from config.yaml")
        
        # Bind mouse wheel for scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
    
    def save_intuitive_config(self, editor_window):
        """Save config from intuitive editor and update main GUI"""
        for full_key, widget_data in self.config_widgets.items():
            parts = full_key.split('.')
            target = self.config
            
            # Navigate to the parent of the target key
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                if isinstance(target[part], dict):
                    target = target[part]
                else:
                    self.log_message(f"Warning: Cannot set {full_key} - path structure issue", logging.WARNING)
                    break
            else:
                last = parts[-1]
                
                # Handle different widget types
                if len(widget_data) == 3 and isinstance(widget_data[2], type):  # Range type (lower, upper, value_type)
                    lower_var, upper_var, value_type = widget_data
                    lower_val = lower_var.get().strip()
                    upper_val = upper_var.get().strip()
                    
                    try:
                        if value_type == int:
                            target[last] = [int(float(lower_val)) if lower_val else 0, 
                                          int(float(upper_val)) if upper_val else 0]
                        else:
                            target[last] = [float(lower_val) if lower_val else 0.0, 
                                          float(upper_val) if upper_val else 0.0]
                    except (ValueError, TypeError):
                        target[last] = [0, 0]
                else:  # Standard type (var, value_type)
                    var, value_type = widget_data
                    value = var.get()
                    
                    # Parse and convert based on type
                    if value_type == bool:
                        # Boolean values come as boolean from BooleanVar
                        target[last] = bool(value)
                    elif value_type == int:
                        try:
                            if value == "" or value == "None":
                                target[last] = None
                            else:
                                target[last] = int(float(value))
                        except (ValueError, TypeError):
                            target[last] = 0
                    elif value_type == float:
                        try:
                            if value == "" or value == "None":
                                target[last] = None
                            else:
                                target[last] = float(value)
                        except (ValueError, TypeError):
                            target[last] = 0.0
                    else:
                        # Handle empty strings for None values
                        if value == "" or value == "None":
                            target[last] = None
                        else:
                            target[last] = value
        
        # Update main GUI parameters from config
        self.start_point_var.set(self.config['signal_process']['null_point'])
        self.two_saw_var.set(self.config['lorentzian']['bimodal_fit'])
        self.baseline_var.set(self.config['signal_process']['baseline_correction']['enabled'])
        
        # Also update the grating spacing in the main GUI if changed in editor
        if 'tgs' in self.config and 'grating_spacing' in self.config['tgs']:
            self.grating_edit.delete(0, tk.END)
            self.grating_edit.insert(0, f"{self.config['tgs']['grating_spacing']:.6f}")
        
        # Save to file and log
        self.save_config()
        self.log_message("Configuration updated from editor")
        
        # Close the editor window
        editor_window.destroy()

    def _build_intuitive_config_section(self, parent, section_title, params_config):
        """Build a more intuitive configuration section with better widgets"""
        frame = ttk.LabelFrame(parent, text=section_title, padding=(8, 5))
        frame.pack(fill='x', pady=(0, 8))
        
        for param in params_config:
            row = ttk.Frame(frame, style='Panel.TFrame')
            row.pack(fill='x', pady=3)
            
            # Parameter label with tooltip
            label = ttk.Label(row, text=param['label'], width=24, anchor='e')
            label.pack(side='left', padx=(0, 8))
            self.add_tooltip(label, param['tooltip'])
            
            full_key = param['key']
            current_value = self._get_nested_config_value(full_key)
            
            if param['type'] == 'toggle':
                # Boolean toggle using Checkbutton - handle None or empty values
                if current_value is None:
                    bool_value = param.get('default', False)
                elif isinstance(current_value, bool):
                    bool_value = current_value
                else:
                    # Convert to boolean if it's a string or other type
                    bool_value = bool(current_value) if current_value else False
                
                var = tk.BooleanVar(value=bool_value)
                widget = ttk.Checkbutton(row, text=param.get('on_text', 'Enabled'), variable=var)
                widget.pack(side='left', padx=(0, 5))
                self.config_widgets[full_key] = (var, bool)
                self.add_tooltip(widget, param['tooltip'])
                
            elif param['type'] == 'select':
                # Dropdown selection with dark theme
                if current_value is None:
                    default_value = param.get('default', param['options'][0])
                else:
                    default_value = str(current_value)
                
                var = tk.StringVar(value=default_value)
                widget = ttk.Combobox(row, textvariable=var, values=param['options'], state='readonly', width=20, style='Panel.TCombobox')
                widget.pack(side='left', padx=(0, 5))
                # Store the appropriate type
                target_type = param.get('value_type', str)
                self.config_widgets[full_key] = (var, target_type)
                self.add_tooltip(widget, param['tooltip'])
                
            elif param['type'] == 'range':
                # Two entry boxes for lower and upper bounds
                range_frame = ttk.Frame(row, style='Panel.TFrame')
                range_frame.pack(side='left', padx=(0, 5))
                
                # Lower bound
                lower_label = ttk.Label(range_frame, text="min:", font=('Arial', 8), foreground=self.fg_color)
                lower_label.pack(side='left', padx=(2, 2))
                
                lower_var = tk.StringVar()
                if current_value and isinstance(current_value, list) and len(current_value) >= 1:
                    lower_var.set(str(current_value[0]))
                elif param.get('default'):
                    lower_var.set(str(param['default'][0]))
                else:
                    lower_var.set("0")
                
                lower_entry = ttk.Entry(range_frame, textvariable=lower_var, width=10, style='Panel.TEntry')
                lower_entry.pack(side='left', padx=(0, 5))
                
                # Upper bound
                upper_label = ttk.Label(range_frame, text="max:", font=('Arial', 8), foreground=self.fg_color)
                upper_label.pack(side='left', padx=(2, 2))
                
                upper_var = tk.StringVar()
                if current_value and isinstance(current_value, list) and len(current_value) >= 2:
                    upper_var.set(str(current_value[1]))
                elif param.get('default'):
                    upper_var.set(str(param['default'][1]))
                else:
                    upper_var.set("100")
                
                upper_entry = ttk.Entry(range_frame, textvariable=upper_var, width=10, style='Panel.TEntry')
                upper_entry.pack(side='left')
                
                # Store both variables with a special handler
                self.config_widgets[full_key] = (lower_var, upper_var, param.get('value_type', float))
                self.add_tooltip(lower_entry, param['tooltip'])
                self.add_tooltip(upper_entry, param['tooltip'])
                
            elif param['type'] == 'number':
                # Numeric entry with optional bounds
                if current_value is None:
                    default_value = str(param.get('default', 0))
                else:
                    default_value = str(current_value)
                
                var = tk.StringVar(value=default_value)
                widget = ttk.Entry(row, textvariable=var, width=15, style='Panel.TEntry')
                widget.pack(side='left', padx=(0, 5))
                
                # Add validation for numbers
                value_type = param.get('value_type', float)
                
                def validate_number(P, vt=value_type):
                    if P == "" or P == "-" or P == ".":
                        return True
                    try:
                        if vt == int:
                            int(P)
                        else:
                            float(P)
                        return True
                    except:
                        return False
                
                validate_cmd = row.register(validate_number)
                widget.config(validate='key', validatecommand=(validate_cmd, '%P'))
                
                self.config_widgets[full_key] = (var, value_type)
                self.add_tooltip(widget, param['tooltip'])
                
            elif param['type'] == 'text':
            # Text entry (for paths, names, etc.)
                if current_value is None:
                    default_value = param.get('default', '')
                else:
                    default_value = str(current_value)
                
                var = tk.StringVar(value=default_value)
                widget = ttk.Entry(row, textvariable=var, width=param.get('width', 30), style='Panel.TEntry')
                widget.pack(side='left', fill='x', expand=True, padx=(0, 5))
                self.config_widgets[full_key] = (var, str)
                self.add_tooltip(widget, param['tooltip'])
    
    def _get_nested_config_value(self, key):
        """Get a nested config value by dot-separated key"""
        parts = key.split('.')
        target = self.config
        for part in parts:
            if isinstance(target, dict) and part in target:
                target = target[part]
            else:
                return None
        return target

    def _build_config_section(self, parent, section_path, section_title, params):
        """Build a labeled frame with config parameters"""
        frame = ttk.LabelFrame(parent, text=section_title, padding=(8, 5))
        frame.pack(fill='x', pady=(0, 8))
        
        # Navigate to the config section correctly
        target = self.config
        parts = section_path.split('.')
        
        for part in parts:
            if part in target:
                target = target[part]
            else:
                self.log_message(f"Warning: Config section '{section_path}' not found", logging.WARNING)
                target = {}
                break
        
        if not target and section_path != 'path':
            warn_label = ttk.Label(frame, text=f"Section '{section_title}' not found in config", foreground='#ff6b6b')
            warn_label.pack(pady=5)
            return
        
        for key, tooltip_text in params.items():
            row = ttk.Frame(frame, style='Panel.TFrame')
            row.pack(fill='x', pady=3)
            
            label = ttk.Label(row, text=key, width=22, anchor='e')
            label.pack(side='left', padx=(0, 8))
            self.add_tooltip(label, tooltip_text)
            
            full_key = f"{section_path}.{key}"
            value = target.get(key) if isinstance(target, dict) else None
            
            # Create appropriate widget based on value type
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                widget = ttk.Checkbutton(row, variable=var)
                # Checkbuttons don't need as much width
                widget.pack(side='left', padx=(0, 5))
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    var = tk.DoubleVar(value=value)
                else:
                    var = tk.IntVar(value=value)
                widget = ttk.Entry(row, textvariable=var, width=25)
                widget.pack(side='left', fill='x', expand=True, padx=(0, 5))
            elif isinstance(value, str):
                var = tk.StringVar(value=value)
                # For path strings, use wider entry
                if 'path' in key.lower() or 'filename' in key.lower() or 'name' in key.lower():
                    widget = ttk.Entry(row, textvariable=var, width=40)
                else:
                    widget = ttk.Entry(row, textvariable=var, width=25)
                widget.pack(side='left', fill='x', expand=True, padx=(0, 5))
            elif isinstance(value, list):
                # Convert list to comma-separated string
                display_value = ",".join(str(v) for v in value)
                var = tk.StringVar(value=display_value)
                widget = ttk.Entry(row, textvariable=var, width=25)
                widget.pack(side='left', fill='x', expand=True, padx=(0, 5))
            else:
                # Handle None or other types
                if value is None:
                    var = tk.StringVar(value="")
                    widget = ttk.Entry(row, textvariable=var, width=25)
                    actual_type = str
                else:
                    var = tk.StringVar(value=str(value))
                    widget = ttk.Entry(row, textvariable=var, width=25)
                    actual_type = type(value)
                widget.pack(side='left', fill='x', expand=True, padx=(0, 5))
            
            self.add_tooltip(widget, tooltip_text)
            
            # Store the actual type, not the widget type
            if value is None:
                actual_type = str
            else:
                actual_type = type(value)
            self.config_widgets[full_key] = (var, actual_type)
    
    def _get_nested_config(self, key, target):
        """Get nested config value"""
        if key in target:
            return target[key]
        return None
    
    def save_config_from_editor(self, editor_window):
        """Save config from editor and update main GUI"""
        for full_key, (var, typ) in self.config_widgets.items():
            parts = full_key.split('.')
            target = self.config
            
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                if isinstance(target[part], dict):
                    target = target[part]
                else:
                    self.log_message(f"Warning: Cannot set {full_key} - path structure issue", logging.WARNING)
                    break
            else:
                last = parts[-1]
                value = var.get()
                
                if typ == bool:
                    target[last] = bool(value)
                elif typ == int:
                    try:
                        target[last] = int(float(value))
                    except (ValueError, TypeError):
                        target[last] = 0
                elif typ == float:
                    try:
                        target[last] = float(value)
                    except (ValueError, TypeError):
                        target[last] = 0.0
                elif typ == list:
                    if value and isinstance(value, str):
                        try:
                            target[last] = [float(v.strip()) if '.' in v else int(v.strip()) 
                                        for v in value.split(',') if v.strip()]
                        except ValueError:
                            target[last] = []
                    else:
                        target[last] = []
                else:
                    target[last] = value
        
        self.start_point_var.set(self.config['signal_process']['null_point'])
        self.two_saw_var.set(self.config['lorentzian']['bimodal_fit'])
        self.baseline_var.set(self.config['signal_process']['baseline_correction']['enabled'])
        
        if 'tgs' in self.config and 'grating_spacing' in self.config['tgs']:
            self.grating_edit.delete(0, tk.END)
            self.grating_edit.insert(0, f"{self.config['tgs']['grating_spacing']:.6f}")
        
        self.save_config()
        self.log_message("Configuration updated from editor")
        editor_window.destroy()
    
    def reload_config_to_editor(self, editor_window):
        """Reload config from file and update editor"""
        self.config = self.load_config()
        editor_window.destroy()
        self.open_config_editor()
    
    def toggle_baseline_ui(self):
        """Enable/disable baseline UI elements"""
        state = 'normal' if self.baseline_var.get() else 'disabled'
        self.baseline_button.config(state=state)
    
    def select_calib_files(self):
        """Select calibration files"""
        files = filedialog.askopenfilenames(
            title="Select Calibration files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not files:
            return
        
        pos_file, neg_file = self.match_files(files)
        if pos_file and neg_file:
            self.calib_pos_file = pos_file
            self.calib_neg_file = neg_file
            pos_name = Path(pos_file).stem
            neg_name = Path(neg_file).stem
            self.calib_file_label.config(text=f"P: {pos_name} | N: {neg_name}", foreground=self.fg_color)
        else:
            messagebox.showerror("Error", "No matching POS/NEG pair found.")
    
    def select_baseline_files(self):
        """Select baseline files"""
        files = filedialog.askopenfilenames(
            title="Select Baseline files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not files:
            return
        
        pos_file, neg_file = self.match_files(files)
        if pos_file and neg_file:
            self.baseline_pos_file = pos_file
            self.baseline_neg_file = neg_file
            pos_name = Path(pos_file).stem
            neg_name = Path(neg_file).stem
            self.baseline_file_label.config(text=f"P: {pos_name} | N: {neg_name}", foreground=self.fg_color)
        else:
            messagebox.showerror("Error", "No matching baseline POS/NEG pair found.")
    
    def match_files(self, file_list):
        """Match POS and NEG files from a list"""
        pos_file = None
        neg_file = None
        
        for f in file_list:
            name = Path(f).stem.upper()
            if 'POS' in name:
                pos_file = f
            elif 'NEG' in name:
                neg_file = f
        
        if pos_file and neg_file:
            pos_base = Path(pos_file).stem.upper().replace('POS', '')
            neg_base = Path(neg_file).stem.upper().replace('NEG', '')
            if pos_base != neg_base:
                return None, None
        
        return pos_file, neg_file
    
    def run_calibration(self):
        """Run calibration to determine grating spacing"""
        if not self.calib_pos_file or not self.calib_neg_file:
            messagebox.showerror("Error", "Please select calibration files.")
            return
        
        self.log_message("Starting calibration...")
        
        def calibration_thread():
            try:
                temp_config = {
                    'signal_process': {
                        'heterodyne': 'di-homodyne',
                        'null_point': int(self.start_point_var.get()),
                        'initial_samples': 50,
                        'baseline_correction': {'enabled': False}
                    },
                    'fft': self.config['fft'],
                    'lorentzian': self.config['lorentzian'].copy(),
                    'plot': {'signal_process': False, 'fft_lorentzian': False, 'tgs': False}
                }
                temp_config['lorentzian']['bimodal_fit'] = self.two_saw_var.get()
                
                signal, _, _, _ = process_signal(
                    temp_config, None, 0, self.calib_pos_file, self.calib_neg_file, 
                    grating_spacing=1.0, **temp_config['signal_process']
                )
                
                time = signal[:, 0]
                amp = signal[:, 1]
                dt = time[1] - time[0]
                derivative = np.gradient(amp, dt)
                saw_signal = np.column_stack((time[:-1], derivative[:-1]))
                
                fft_signal = fft(saw_signal, **temp_config['fft'])
                
                # lorentzian_fit now returns 10 values, we only need the first 2 (f, f_err)
                # Unpack all values but only use what we need
                result = lorentzian_fit(
                    temp_config, None, 0, fft_signal, **temp_config['lorentzian']
                )
                
                # Extract f and f_err from the 10-value tuple
                f = result[0]
                f_err = result[1]
                # The rest: fwhm, tau, snr, frequency_bounds, fit_function, popt, fft_segment, fft_full, lorentzian_curve = result[2:]
                
                if isinstance(f, np.ndarray):
                    f = f[0]
                grating_spacing_um = (self.sound_speed / f) * 1e6
                self.calibrated_spacing = grating_spacing_um
                
                self.root.after(0, lambda: self.update_calibration_result(f, grating_spacing_um))
                self.log_message(f"Calibration: f = {f/1e6:.3f} MHz, grating = {grating_spacing_um:.4f} µm")
                
                if self.close_plots_var.get():
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Calibration Error", str(e)))
                self.log_message(f"Calibration error: {str(e)}", logging.ERROR)
        
        thread = threading.Thread(target=calibration_thread)
        thread.daemon = True
        thread.start()
    
    def update_calibration_result(self, frequency, grating_spacing):
        """Update UI with calibration results"""
        self.grating_edit.delete(0, tk.END)
        self.grating_edit.insert(0, f"{grating_spacing:.6f}")
    
    def browse_log_file(self):
        """Browse for log file location"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.log_file_var.set(file_path)
    
    def add_batch_files(self):
        """Add files to batch queue"""
        files = filedialog.askopenfilenames(
            title="Select TGS files for batch processing",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not files:
            return
        
        pos_neg_pairs = {}
        for f in files:
            name = Path(f).stem
            base = name.upper().replace('POS', '').replace('NEG', '')
            if 'POS' in name.upper():
                pos_neg_pairs[base] = {'pos': f, 'neg': pos_neg_pairs.get(base, {}).get('neg')}
            elif 'NEG' in name.upper():
                pos_neg_pairs[base] = {'neg': f, 'pos': pos_neg_pairs.get(base, {}).get('pos')}
        
        added = 0
        for base, pair in pos_neg_pairs.items():
            if pair.get('pos') and pair.get('neg'):
                self.pos_files.append(pair['pos'])
                self.neg_files.append(pair['neg'])
                pos_name = Path(pair['pos']).stem
                neg_name = Path(pair['neg']).stem
                self.batch_listbox.insert(tk.END, f"[{len(self.pos_files)}] P: {pos_name} | N: {neg_name}")
                added += 1
        
        self.log_message(f"Added {added} file pairs to queue")
        if added == 0:
            messagebox.showwarning("Warning", "No matching POS/NEG pairs found.")
    
    def clear_queue(self):
        """Clear the batch queue"""
        self.pos_files = []
        self.neg_files = []
        self.batch_listbox.delete(0, tk.END)
        self.file_to_plot_path.clear()
        self.file_to_fit_params.clear()
        if hasattr(self, 'file_to_fit_plot_data'):
            self.file_to_fit_plot_data.clear()
        self.clear_results_table()
        self.log_message("Queue cleared")
    
    def run_batch(self):
        """Run batch processing"""
        grating_text = self.grating_edit.get().strip()
        if not grating_text or float(grating_text) <= 0:
            messagebox.showerror("Error", "Please run calibration or enter a valid grating spacing.")
            return
        
        # Check baseline files if baseline is enabled
        if self.baseline_var.get():
            if not self.baseline_pos_file or not self.baseline_neg_file:
                messagebox.showerror("Error", "Baseline correction is enabled but no baseline files selected.\n\nPlease either:\n1. Select baseline files using the 'Select baseline' button, or\n2. Disable 'Use baseline' checkbox.")
                return

        if not self.pos_files:
            messagebox.showerror("Error", "No files in queue. Please add files first.")
            return
        
        null_point = self.start_point_var.get()
        if null_point < 1 or null_point > 4:
            self.log_message(f"Warning: null_point={null_point} is invalid, setting to 2")
            null_point = 2
            self.start_point_var.set(2)
        
        self.config['signal_process']['null_point'] = int(null_point)
        self.config['lorentzian']['bimodal_fit'] = bool(self.two_saw_var.get())
        self.config['signal_process']['baseline_correction']['enabled'] = bool(self.baseline_var.get())
        self.config['tgs']['grating_spacing'] = float(grating_text)
        
        if 'dc_filter_range' in self.config['lorentzian']:
            dc_range = self.config['lorentzian']['dc_filter_range']
            if isinstance(dc_range, list) and len(dc_range) == 2:
                self.config['lorentzian']['dc_filter_range'] = [int(dc_range[0]), int(dc_range[1])]
        
        if self.baseline_var.get() and self.baseline_pos_file and self.baseline_neg_file:
            self.config['signal_process']['baseline_correction']['pos'] = self.baseline_pos_file
            self.config['signal_process']['baseline_correction']['neg'] = self.baseline_neg_file
        
        show_plots = not self.close_plots_var.get()
        self.config['plot']['signal_process'] = show_plots
        self.config['plot']['fft_lorentzian'] = show_plots
        self.config['plot']['tgs'] = show_plots
        
        if self.config['plot']['settings']['num_points'] is None or self.config['plot']['settings']['num_points'] == 'None':
            self.config['plot']['settings']['num_points'] = 10000
        
        data_dir = Path(self.pos_files[0]).parent
        self.config['path'] = str(data_dir)
        
        log_path = self.log_file_var.get().strip()
        if not log_path:
            first_file = Path(self.pos_files[0])
            log_path = first_file.parent / f"{first_file.stem.split('-POS')[0]}_postprocessing.txt"
        else:
            log_path = Path(log_path)
            if log_path.suffix == '':
                log_path = log_path.with_suffix('.txt')
            log_path = str(log_path)
        
        self.current_results_log_path = log_path
        if not self.log_file_var.get().strip():
            self.log_file_var.set(str(log_path))
        
        self.save_config()
        self.log_message(f"Starting batch processing of {len(self.pos_files)} file pairs...")
        self.log_message(f"Plot saving: {'ENABLED' if show_plots else 'DISABLED'}")
        self.log_message(f"Results will be saved to: {log_path}")
        
        # Configure progress bar
        self.progress.configure(mode='determinate', maximum=len(self.pos_files), value=0)
        self.progress['value'] = 0  # Explicitly set value
        self.status_var.set("Starting batch... (0/{})".format(len(self.pos_files)))
        
        self.stop_batch = False
        
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        self.file_to_plot_path.clear()
        self.file_to_fit_params.clear()
        
        # Initialize time tracking
        self.batch_start_time = None
        self.first_run_duration = None
        self.current_run_start_time = None
        self.status_var.set("Starting batch...")
        
        def batch_thread():
            try:
                self._run_direct_batch(log_path)
                if not self.stop_batch:
                    self.root.after(0, self._batch_finished, True, "Batch processing completed.")
                else:
                    self.root.after(0, self._batch_finished, True, "Batch processing stopped by user.")
            except Exception as e:
                self.root.after(0, self._batch_finished, False, str(e))
        
        thread = threading.Thread(target=batch_thread)
        thread.daemon = True
        self.running_job = thread
        thread.start()
    
    def update_status_with_time(self, current_file_index, total_files):
        """Update the status message with estimated time remaining"""
        import time
        
        # For the first file, just show estimating
        if current_file_index == 1:
            if self.batch_start_time is None:
                self.batch_start_time = time.time()
            self.status_var.set(f"Processing file 1/{total_files}... (estimating...)")
            self.root.update_idletasks()
            return
        
        # Check if batch_start_time is valid
        if self.batch_start_time is None:
            self.status_var.set(f"Processing file {current_file_index}/{total_files}...")
            self.root.update_idletasks()
            return
        
        # Calculate duration so far
        elapsed_time = time.time() - self.batch_start_time
        
        # Calculate average time per file based on completed files
        completed_files = current_file_index - 1
        if completed_files > 0:
            avg_time_per_file = elapsed_time / completed_files
            
            # Calculate remaining files
            remaining_files = total_files - current_file_index + 1
            estimated_remaining = avg_time_per_file * remaining_files
            
            if estimated_remaining > 0:
                if estimated_remaining < 60:
                    time_str = f"{estimated_remaining:.0f} seconds"
                elif estimated_remaining < 3600:
                    minutes = int(estimated_remaining // 60)
                    seconds = int(estimated_remaining % 60)
                    time_str = f"{minutes} min {seconds} sec"
                else:
                    hours = int(estimated_remaining // 3600)
                    minutes = int((estimated_remaining % 3600) // 60)
                    time_str = f"{hours} hr {minutes} min"
                
                self.status_var.set(f"Processing file {current_file_index}/{total_files}... ~{time_str} remaining")
            else:
                self.status_var.set(f"Processing file {current_file_index}/{total_files}...")
        else:
            self.status_var.set(f"Processing file {current_file_index}/{total_files}...")
        
        self.root.update_idletasks()

    def stop_batch_processing(self):
        """Stop the batch processing gracefully"""
        if hasattr(self, 'running_job') and self.running_job and self.running_job.is_alive():
            self.stop_batch = True
            self.log_message("Stopping batch processing... (will finish current file)")
            self.status_var.set("Stopping... (finishing current file)")
            # Disable stop button to prevent multiple clicks
            self.stop_button.config(state='disabled')

    def _run_direct_batch(self, log_path):
        """Run batch directly using tgs_fit"""
        from src.analysis.tgs import tgs_fit
        import matplotlib.pyplot as plt
        import copy
        import gc
        
        plt.close('all')
        gc.collect()
        
        self.root.after(0, self.clear_preview)

        results = []
        total = len(self.pos_files)
        
        data_dir = Path(self.pos_files[0]).parent
        
        from src.core.path import Paths
        paths = Paths(
            data_dir=data_dir,
            figure_dir=data_dir / 'figures',
            fit_dir=data_dir / 'fit',
            fit_path=data_dir / 'fit' / 'fit.csv',
            signal_path=data_dir / 'fit' / 'signal.json',
        )
        paths.figure_dir.mkdir(parents=True, exist_ok=True)
        paths.fit_dir.mkdir(parents=True, exist_ok=True)
        
        show_plots = not self.close_plots_var.get()
        self.config['plot']['signal_process'] = show_plots
        self.config['plot']['fft_lorentzian'] = show_plots
        self.config['plot']['tgs'] = show_plots
        
        if self.config['plot']['settings']['num_points'] is None or self.config['plot']['settings']['num_points'] == 'None':
            self.config['plot']['settings']['num_points'] = 10000
        
        grating_spacing_val = float(self.grating_edit.get())
        
        import time
        
        for i, (pos_file, neg_file) in enumerate(zip(self.pos_files, self.neg_files), 1):
            if self.stop_batch:
                self.log_message("Batch processing stopped by user.")
                break
            
            file_id = Path(pos_file).stem
            self.log_message(f"Processing [{i}/{total}]: {file_id}")
            
            # Update status with time estimate - pass current file index
            def update_progress(val, tot):
                self.progress.configure(value=val)
                self.root.update_idletasks()

            self.root.after(0, update_progress, i, total)

            # Update status with time estimate for current file
            self.root.after(0, lambda: self.update_status_with_time(i, total))
            
            try:
                file_config = copy.deepcopy(self.config)
                
                file_config['signal_process']['null_point'] = int(file_config['signal_process']['null_point'])
                file_config['signal_process']['initial_samples'] = int(file_config['signal_process']['initial_samples'])
                file_config['lorentzian']['dc_filter_range'] = [
                    int(file_config['lorentzian']['dc_filter_range'][0]),
                    int(file_config['lorentzian']['dc_filter_range'][1])
                ]
                
                if file_config['plot']['settings']['num_points'] is None or file_config['plot']['settings']['num_points'] == 'None':
                    file_config['plot']['settings']['num_points'] = 10000
                else:
                    file_config['plot']['settings']['num_points'] = int(file_config['plot']['settings']['num_points'])
                
                (start_idx, start_time, grating_spacing, 
                A, A_err, B, B_err, C, C_err, 
                alpha, alpha_err, beta, beta_err, 
                theta, theta_err, tau, tau_err, 
                f, f_err, signal, fft_full, lorentzian_curve) = tgs_fit(
                    file_config, paths, i, str(pos_file), str(neg_file), 
                    grating_spacing=grating_spacing_val,
                    signal_proportion=float(file_config['tgs']['signal_proportion']),
                    maxfev=int(file_config['tgs']['maxfev'])
                )

                plt.close('all')

                date_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                
                # Extract values correctly
                # A, B, C are already scalar values from tgs_fit
                # alpha, beta, theta, tau, f are scalar values
                
                # Store fit parameters for this file (directly from tgs_fit outputs)
                fit_params = {
                    'A': float(A) if A is not None else None,
                    'A_err': float(A_err) if A_err is not None else None,
                    'B': float(B) if B is not None else None,
                    'B_err': float(B_err) if B_err is not None else None,
                    'C': float(C) if C is not None else None,
                    'C_err': float(C_err) if C_err is not None else None,
                    'alpha': float(alpha) if alpha is not None else None,
                    'alpha_err': float(alpha_err) if alpha_err is not None else None,
                    'beta': float(beta) if beta is not None else None,
                    'beta_err': float(beta_err) if beta_err is not None else None,
                    'theta': float(theta) if theta is not None else None,
                    'theta_err': float(theta_err) if theta_err is not None else None,
                    'tau': float(tau) if tau is not None else None,
                    'tau_err': float(tau_err) if tau_err is not None else None,
                    'f': float(f) if f is not None else None,
                    'f_err': float(f_err) if f_err is not None else None,
                }

                # Also store the actual signal data for plotting
                # Create fit data dictionary for interactive plotting
                fit_data_for_plot = {
                    'title': file_id,
                    'time_raw': signal[:, 0],
                    'signal_raw': signal[:, 1],
                }

                # Generate time points for fit
                time_fit = np.linspace(signal[start_idx, 0], signal[-1, 0], 1000)
                from src.analysis.functions import tgs_function
                functional_func, thermal_func = tgs_function(start_time, grating_spacing_val)

                fit_data_for_plot['time_fit'] = time_fit
                fit_data_for_plot['fit_signal'] = functional_func(time_fit, A, B, C, alpha, beta, theta, tau, f)
                fit_data_for_plot['thermal_signal'] = thermal_func(time_fit, A, B, C, alpha, beta, theta, tau, f)

                # Store the actual FFT data and Lorentzian fit
                if fft_full is not None and len(fft_full) > 0:
                    self.log_message(f"  FFT data shape: {fft_full.shape}", logging.DEBUG)
                    fit_data_for_plot['fft_freq'] = fft_full[:, 0]  # Frequency in GHz
                    fit_data_for_plot['fft_amp'] = fft_full[:, 1]   # Amplitude
                else:
                    self.log_message(f"  WARNING: fft_full is None or empty", logging.INFO)
                    fit_data_for_plot['fft_freq'] = None
                    fit_data_for_plot['fft_amp'] = None

                # Store Lorentzian fit curve if available
                if lorentzian_curve is not None and len(lorentzian_curve) > 0:
                    # Use the frequency range from the FFT data for the Lorentzian curve
                    if fft_full is not None and len(fft_full) > 0:
                        # Get FFT frequencies (in GHz) and amplitudes
                        fft_freqs_ghz = fft_full[:, 0]  # Already in GHz
                        fft_amps = fft_full[:, 1]
                        
                        # Get frequency bounds from config (in GHz)
                        freq_bounds = self.config['lorentzian'].get('frequency_bounds', [0.1, 0.9])
                        lorentzian_freqs_ghz = np.linspace(freq_bounds[0], freq_bounds[1], len(lorentzian_curve))
                        
                        # Scale Lorentzian curve to match FFT peak amplitude
                        mask = (fft_freqs_ghz >= freq_bounds[0]) & (fft_freqs_ghz <= freq_bounds[1])
                        if np.any(mask):
                            fft_peak_in_range = np.max(fft_amps[mask])
                        else:
                            fft_peak_in_range = np.max(fft_amps)
                        
                        lorentzian_peak_value = np.max(lorentzian_curve)
                        if lorentzian_peak_value > 0 and fft_peak_in_range > 0:
                            scale_factor = fft_peak_in_range / lorentzian_peak_value
                            lorentzian_curve_scaled = lorentzian_curve * scale_factor
                        else:
                            lorentzian_curve_scaled = lorentzian_curve
                        
                        # Store for plotting - use GHz frequencies
                        fit_data_for_plot['lorentzian_freq'] = lorentzian_freqs_ghz
                        fit_data_for_plot['lorentzian_fit'] = lorentzian_curve_scaled
                    else:
                        fit_data_for_plot['lorentzian_fit'] = None
                else:
                    fit_data_for_plot['lorentzian_fit'] = None

                # Store the fit data for later display
                self.file_to_fit_plot_data[file_id] = fit_data_for_plot
                self.file_to_fit_params[file_id] = fit_params

                # Update the preview and results table with the latest fit (on the main thread)
                self.root.after(0, lambda: self.create_interactive_plot(fit_data_for_plot))
                self.root.after(0, lambda: self.update_results_table(fit_params))
                
                # For results file
                if isinstance(f, np.ndarray):
                    f_val = float(f[0]) if len(f) > 0 else 0.0
                    f_err_val = float(f_err[0]) if len(f_err) > 0 else 0.0
                    tau_val = float(tau[0]) if len(tau) > 0 else 0.0
                    tau_err_val = float(tau_err[0]) if len(tau_err) > 0 else 0.0
                else:
                    f_val = float(f) if f is not None else 0.0
                    f_err_val = float(f_err) if f_err is not None else 0.0
                    tau_val = float(tau) if tau is not None else 0.0
                    tau_err_val = float(tau_err) if tau_err is not None else 0.0
                
                result = {
                    'run_name': file_id,
                    'date_time': date_time_str,
                    'grating_spacing_um': float(grating_spacing),
                    'SAW_freq_Hz': f_val,
                    'SAW_freq_error_Hz': f_err_val,
                    'A_Wm-2': float(A) if A is not None else 0.0,
                    'A_err_Wm-2': float(A_err) if A_err is not None else 0.0,
                    'alpha_m2s-1': float(alpha) if alpha is not None else 0.0,
                    'alpha_err_m2s-1': float(alpha_err) if alpha_err is not None else 0.0,
                    'beta_s0.5': float(beta) if beta is not None else 0.0,
                    'beta_err_s0.5': float(beta_err) if beta_err is not None else 0.0,
                    'B_Wm-2': float(B) if B is not None else 0.0,
                    'B_err_Wm-2': float(B_err) if B_err is not None else 0.0,
                    'theta_rad': float(theta) if theta is not None else 0.0,
                    'theta_err_rad': float(theta_err) if theta_err is not None else 0.0,
                    'tau_s': tau_val,
                    'tau_err_s': tau_err_val,
                    'C_Wm-2': float(C) if C is not None else 0.0,
                    'C_err_Wm-2': float(C_err) if C_err is not None else 0.0,
                }
                
                results.append(result)
                if len(results) % 10 == 0:
                    self.root.after(0, lambda ddir=data_dir: self.cleanup_old_plots(ddir))
                self.log_message("  SUCCESS!")

                # After successful completion, update status for next file (or completion)
                self.root.after(0, lambda: self.update_status_with_time(i, total))

                # Only store the image path as fallback, but don't display it
                current_data_dir = Path(pos_file).parent
                combined_plot_path = current_data_dir / 'figures' / 'combined' / f'combined-{file_id}.png'
                if combined_plot_path.exists():
                    self.file_to_plot_path[file_id] = str(combined_plot_path)

                # Don't automatically display during batch - let user click to view
                
            except Exception as e:
                self.log_message(f"  FAILED: {str(e)}", logging.ERROR)
            
            if self.close_plots_var.get():
                plt.close('all')
        
        if results:
            self._save_space_delimited(results, log_path)
            self.log_message(f"Results saved to {log_path}")
        else:
            self.log_message("No successful fits to save.", logging.WARNING)
    
    def _save_space_delimited(self, results, log_path):
        """Save results as space-delimited file with header"""
        
        if not results:
            return
        
        headers = [
            'run_name', 'date_time', 'grating_spacing_um', 
            'SAW_freq_Hz', 'SAW_freq_error_Hz',
            'A_Wm-2', 'A_err_Wm-2', 'alpha_m2s-1', 'alpha_err_m2s-1',
            'beta_s0.5', 'beta_err_s0.5', 'B_Wm-2', 'B_err_Wm-2',
            'theta_rad', 'theta_err_rad', 'tau_s', 'tau_err_s',
            'C_Wm-2', 'C_err_Wm-2'
        ]
        
        with open(log_path, 'w') as f:
            f.write(' '.join(headers) + '\n')
            
            for result in results:
                row = []
                for header in headers:
                    value = result.get(header, '')
                    if isinstance(value, str):
                        row.append(value)
                    else:
                        if isinstance(value, float):
                            row.append(f"{value:.8e}")
                        else:
                            row.append(str(value))
                f.write(' '.join(row) + '\n')
    
    def clear_preview(self):
        """Clear the preview display"""
        self.preview_ax.clear()
        self.preview_ax.set_facecolor(self.bg_color)
        self.preview_ax.text(0.5, 0.5, 'Batch processing in progress...\nWaiting for first fit',
                            ha='center', va='center', transform=self.preview_ax.transAxes,
                            color=self.fg_color, fontsize=12, fontfamily='Arial')
        self.preview_ax.set_xlim(0, 1)
        self.preview_ax.set_ylim(0, 1)
        self.preview_ax.set_xticks([])
        self.preview_ax.set_yticks([])
        self.preview_canvas.draw()
        self.clear_results_table()

    def _batch_finished(self, success, message):
        """Handle batch completion"""
        self.progress.stop()
        self.progress.configure(mode='indeterminate')
        self.progress.configure(value=0)
        
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if success:
            self.status_var.set("Completed")
            self.log_message(message)
        else:
            self.status_var.set("Failed")
            self.log_message(message, logging.ERROR)
            messagebox.showerror("Error", f"Batch processing failed:\n{message}")
        
        # Reset time tracking
        self.batch_start_time = None
        self.first_run_duration = None
        self.current_run_start_time = None
    
    def clear_log(self):
        """Clear the log text widget"""
        self.log_text.delete(1.0, tk.END)
    
    def on_close(self):
        """Clean up on exit"""
        if self.running_job and self.running_job.is_alive():
            if messagebox.askyesno("Confirm", "A job is still running. Exit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    
    try:
        if getattr(sys, 'frozen', False):
            possible_paths = [
                os.path.join(os.path.dirname(sys.executable), 'dihomodyne_beams_icon.ico'),
                os.path.join(sys._MEIPASS, 'dihomodyne_beams_icon.ico'),
                'dihomodyne_beams_icon.ico'
            ]
        else:
            possible_paths = ['dihomodyne_beams_icon.ico']
        
        icon_path = None
        for path in possible_paths:
            if os.path.exists(path):
                icon_path = path
                break
        
        if icon_path:
            root.iconbitmap(icon_path)
            try:
                icon_image = tk.PhotoImage(file=icon_path)
                root.iconphoto(True, icon_image)
            except:
                pass
    except:
        pass
    
    app = TGSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()