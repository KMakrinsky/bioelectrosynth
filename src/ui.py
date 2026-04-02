import sys
import os
import numpy as np
import scipy.signal as signal
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFormLayout, QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, 
                             QCheckBox, QPushButton, QFileDialog, QScrollArea, QSplitter,
                             QTabWidget, QComboBox, QTextBrowser, QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from .compositor import SignalCompositor
from .circuit_widget import CircuitBuilderWidget
from .study_tab import StudyTab
from .optimization_tab import OptimizationTab

# Try to import WebEngine, fallback to TextBrowser if missing
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

import markdown
import os
import tempfile

class HelpWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Controls area
        btn_layout = QHBoxLayout()
        
        self.btn_open_external = QPushButton("Open in Browser")
        self.btn_open_external.setToolTip("Open in default system browser (Chrome/Edge)")
        self.btn_open_external.clicked.connect(self.open_external)
        
        self.btn_toggle_view = QPushButton("Switch to Simple Text")
        self.btn_toggle_view.setCheckable(True)
        self.btn_toggle_view.toggled.connect(self.toggle_view_mode)
        
        btn_layout.addWidget(self.btn_open_external)
        btn_layout.addWidget(self.btn_toggle_view)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        if HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            layout.addWidget(self.web_view)
        else:
            self.web_view = None
            
        self.text_view = QTextBrowser()
        self.text_view.setOpenExternalLinks(True)
        layout.addWidget(self.text_view)
        
        if HAS_WEBENGINE:
            self.text_view.hide()
        else:
            self.text_view.show()
            self.btn_toggle_view.setEnabled(False)
            self.btn_toggle_view.setText("Simple Text Only")
        
        # Determine paths
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.getcwd()
            
        md_path = os.path.join(base_path, 'CALCULATION_LOGIC.md')
        
        self.content_md = "# Error\nHelp file not found."
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                self.content_md = f.read()

        if HAS_WEBENGINE:
            self.set_html_content(self.content_md, base_path)
            
        self.text_view.setMarkdown(self.content_md)

    def toggle_view_mode(self, checked):
        if not HAS_WEBENGINE:
            return
            
        if checked:
            self.btn_toggle_view.setText("Switch to Web View")
            if self.web_view: self.web_view.hide()
            self.text_view.show()
        else:
            self.btn_toggle_view.setText("Switch to Simple Text")
            self.text_view.hide()
            if self.web_view: self.web_view.show()

    def set_html_content(self, md_text, base_path):
        # Convert Markdown to HTML with MathJax support extensions
        try:
            html_body = markdown.markdown(
                md_text, 
                extensions=[
                    'pymdownx.arithmatex', 
                    'markdown.extensions.tables', 
                    'markdown.extensions.fenced_code'
                ],
                extension_configs={
                    'pymdownx.arithmatex': {
                        'generic': True,
                        # Ensure $$...$$ blocks are treated as math (prevents Markdown from
                        # interpreting underscores/pipes inside formulas).
                        'block_syntax': ['dollar', 'begin'],
                        'inline_syntax': ['dollar'],
                    }
                }
            )
        except ImportError:
            # Fallback if extensions missing
            html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

        # MathJax Configuration
        # We try to load local MathJax script for offline support.
        # If not found, we fallback to CDN (requires internet).
        
        mathjax_js_content = ""
        # 1. Try embedded file (PyInstaller)
        if getattr(sys, 'frozen', False):
             mj_path = os.path.join(sys._MEIPASS, 'mathjax.js')
        else:
             # 2. Try local source file (check root and src)
             # We moved mathjax.js to root, so let's look there first
             root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             mj_path = os.path.join(root_path, 'mathjax.js')
             if not os.path.exists(mj_path):
                 # Fallback to src
                 mj_path = os.path.join(os.path.dirname(__file__), 'mathjax.js')

        if os.path.exists(mj_path):
             with open(mj_path, 'r', encoding='utf-8') as f:
                 mathjax_js_content = f.read()
             mathjax_script = f"<script>{mathjax_js_content}</script>"
        else:
             # Fallback to CDN
             mathjax_script = '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>'

        # Configure MathJax to process standard delimiters
        mathjax_config = """
        <script>
        window.MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true
          },
          options: {
            ignoreHtmlClass: 'tex2jax_ignore',
            processHtmlClass: 'tex2jax_process'
          }
        };
        </script>
        """

        # Construct full HTML with safe string substitution for scripts (avoiding f-string issues with braces)
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            {config}
            {script}
            <style>
                html {{ font-size: 14pt; }}
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 20px; font-size: 1rem; line-height: 1.5; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eaeaea; padding-bottom: 10px; }}
                code {{ background-color: #f8f9fa; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
                pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                mjx-container[jax="SVG"]:not([display="true"]) {{ display: inline-block; vertical-align: -0.1em; }}
                mjx-container[jax="SVG"][display="true"] {{ display: block; overflow-x: auto; overflow-y: hidden; padding: 6px 0; }}
                .arithmatex {{ overflow-x: auto; }}
                .math-display {{ overflow-x: auto; }}
            </style>
        </head>
        <body>
            {body}
        </body>
        </html>
        """
        
        full_html = html_template.format(
            config=mathjax_config,
            script=mathjax_script,
            body=html_body
        )
        
        # Save to temporary file to avoid URL scheme issues and blank pages
        self.temp_file = os.path.join(tempfile.gettempdir(), 'bioelectrosynth_help.html')
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
        self.web_view.setUrl(QUrl.fromLocalFile(self.temp_file))

    def open_external(self):
        # Create a temp HTML file if simpler view is needed, or just open the MD file if user has viewer?
        # Better to open the HTML we generated (if webengine) or just the text.
        
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.temp_file))
        else:
            # Fallback: create a temp html file from MD
            tmp = os.path.join(tempfile.gettempdir(), 'bioelectrosynth_manual.html')
            # minimal html
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(f"<pre>{self.content_md}</pre>")
            QDesktopServices.openUrl(QUrl.fromLocalFile(tmp))

class ElectrodeWidget(pg.PlotWidget):
    """
    Interactive widget to position electrodes on a mouse schematic.
    """
    sigElectrodesMoved = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setTitle("Electrode Placement")
        # Explicitly set background to white to ensure schematic visibility
        self.setBackground('w')
        self.setXRange(0, 1)
        self.setYRange(0, 1)
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        
        # Mouse Body Size Scale (approx 5 cm length for torso)
        self.MOUSE_SIZE_CM = 5.0
        
        # Draw Mouse Schematic (Simplified as shapes)
        # Body (Ellipse)
        theta = np.linspace(0, 2*np.pi, 100)
        x_body = 0.5 + 0.2 * np.cos(theta)
        y_body = 0.5 + 0.3 * np.sin(theta)
        self.plot(x_body, y_body, pen=pg.mkPen(color='k', width=2))
        
        # Head
        x_head = 0.5 + 0.1 * np.cos(theta)
        y_head = 0.85 + 0.1 * np.sin(theta)
        self.plot(x_head, y_head, pen=pg.mkPen(color='k', width=2))
        
        # Heart Location (Red Cross)
        self.plot([0.5], [0.55], symbol='+', symbolSize=15, symbolPen='r', name="Heart")
        
        # Electrodes (Draggable Points)
        self.working_electrode = pg.ScatterPlotItem(size=15, pen=pg.mkPen(width=2, color='b'), brush=pg.mkBrush(0, 0, 255, 150))
        self.working_electrode.addPoints([{'pos': (0.4, 0.4), 'data': 'Working'}])
        self.addItem(self.working_electrode)
        
        self.counter_electrode = pg.ScatterPlotItem(size=15, pen=pg.mkPen(width=2, color='g'), brush=pg.mkBrush(0, 255, 0, 150))
        self.counter_electrode.addPoints([{'pos': (0.6, 0.4), 'data': 'Counter'}])
        self.addItem(self.counter_electrode)
        
        # Scale Bar (Linear ruler)
        # 1 unit in plot = MOUSE_SIZE_CM (5.0 cm) approx? 
        # No, the plot is 0..1. The body is ~0.6 units high (0.2 to 0.8).
        # If body is 5cm, then 0.6 units = 5 cm. So 1 unit = 5 / 0.6 = 8.33 cm.
        # Let's draw a 1 cm bar.
        # Length in plot units = 1.0 / (self.MOUSE_SIZE_CM / 0.6) = 0.6 / 5.0 = 0.12 units
        bar_len = 0.12 
        bar_x = [0.8, 0.8 + bar_len]
        bar_y = [0.1, 0.1]
        self.plot(bar_x, bar_y, pen=pg.mkPen(color='k', width=3))
        self.text_scale = pg.TextItem("1 cm", color='k', anchor=(0.5, 0))
        self.text_scale.setPos(0.8 + bar_len/2, 0.08)
        self.addItem(self.text_scale)
        
        # Labels
        self.lbl_w = pg.TextItem("W", color='b')
        self.lbl_w.setPos(0.4, 0.4)
        self.addItem(self.lbl_w)
        
        self.lbl_c = pg.TextItem("C/R", color='g')
        self.lbl_c.setPos(0.6, 0.4)
        self.addItem(self.lbl_c)
        
        # Interaction State
        self.dragging = None
        
        # Connect mouse events manually since ScatterPlotItem sigClicked is sometimes tricky with dragging logic
        # We'll use the scene events
        self.scene().sigMouseClicked.connect(self.on_click)
        self.scene().sigMouseMoved.connect(self.on_move)
        
    def on_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            mouse_point = self.plotItem.vb.mapSceneToView(pos)
            
            # Check distance to points
            # Use safe access as in get_positions
            w_pts = self.working_electrode.points()
            if w_pts:
                p = w_pts[0].pos()
                w_pos = np.array([p.x(), p.y()])
            else:
                w_pos = np.array([0.4, 0.4])
                
            c_pts = self.counter_electrode.points()
            if c_pts:
                p = c_pts[0].pos()
                c_pos = np.array([p.x(), p.y()])
            else:
                c_pos = np.array([0.6, 0.4])
            
            m_pos = np.array([mouse_point.x(), mouse_point.y()])
            
            if np.linalg.norm(m_pos - w_pos) < 0.05:
                self.dragging = 'Working'
            elif np.linalg.norm(m_pos - c_pos) < 0.05:
                self.dragging = 'Counter'
            else:
                self.dragging = None
                
        elif event.button() == Qt.MouseButton.RightButton: # Reset drag on right click
             self.dragging = None

    def on_move(self, pos):
        if self.dragging:
            mouse_point = self.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            # Clamp to [0, 1]
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            
            if self.dragging == 'Working':
                self.working_electrode.setData(pos=[(x, y)])
                self.lbl_w.setPos(x, y)
                self.sigElectrodesMoved.emit()
            elif self.dragging == 'Counter':
                self.counter_electrode.setData(pos=[(x, y)])
                self.lbl_c.setPos(x, y)
                self.sigElectrodesMoved.emit()

    def get_positions(self):
        # Access points safely
        # ScatterPlotItem.data returns a structured array or list of dicts depending on version
        # But here we used addPoints with 'pos' in dict.
        # Let's check how pyqtgraph stores it internally. 
        # Usually it's in self.working_electrode.data which is a numpy structured array
        
        # Safe way for ScatterPlotItem with 1 point:
        # Get the underlying data array
        pts_w = self.working_electrode.points()
        if pts_w:
            w_pos = pts_w[0].pos()
            w = (w_pos.x(), w_pos.y())
        else:
            w = (0.4, 0.4)
            
        pts_c = self.counter_electrode.points()
        if pts_c:
            c_pos = pts_c[0].pos()
            c = (c_pos.x(), c_pos.y())
        else:
            c = (0.6, 0.4)
            
        return {"working": w, "counter": c}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioElectroSynth: Magnesium Implant Simulator")
        self.resize(1200, 800)
        
        self.compositor = SignalCompositor()
        self.current_data = None
        # Domain of the currently displayed primary signal:
        # - "A": current (Amps)  [default in this app, ZRA/shunt measurement]
        # - "V": voltage (Volts) [optional secondary channel]
        self.current_domain = "A"
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Left Sidebar (Controls) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(380)
        
        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(controls_widget)
        
        # 1. Global Settings
        self.create_global_controls()
        
        # 2. Corrosion Settings
        self.create_corrosion_controls()
        
        # 3. Bio Settings
        self.create_bio_controls()
        
        # 4. Sensor Settings
        self.create_sensor_controls()
        
        # 5. Electrode Placement (New)
        self.electrode_widget = ElectrodeWidget()
        self.electrode_widget.setFixedHeight(200)
        self.electrode_widget.sigElectrodesMoved.connect(self.recalc_rs)
        self.controls_layout.addWidget(self.electrode_widget)
        
        # Generate Button
        self.btn_generate = QPushButton("Generate Signal")
        self.btn_generate.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_generate.clicked.connect(self.generate_signal)
        self.controls_layout.addWidget(self.btn_generate)

        # Load Experimental CSV Button
        self.btn_load_csv = QPushButton("Load Experimental CSV...")
        self.btn_load_csv.setToolTip("Load an experimental recording and compute PSD/Chebyshev spectrum")
        self.btn_load_csv.clicked.connect(self.load_experimental_csv)
        self.controls_layout.addWidget(self.btn_load_csv)
        
        # Export Button
        self.btn_export = QPushButton("Export Data...")
        self.btn_export.clicked.connect(self.export_data)
        self.controls_layout.addWidget(self.btn_export)

        # Reset Zoom Button
        self.btn_reset_zoom = QPushButton("Reset Zoom")
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)
        self.controls_layout.addWidget(self.btn_reset_zoom)
        
        self.controls_layout.addStretch()
        scroll.setWidget(controls_widget)
        self.left_sidebar = scroll        # keep reference for show/hide
        main_layout.addWidget(scroll)
        
        # --- Right Area (Visualization) ---
        right_layout = QVBoxLayout()
        
        # Tabs for Simulation and Documentation
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        
        # Tab 1: Simulation
        self.sim_tab = QWidget()
        self.tabs.addTab(self.sim_tab, "Simulation")
        
        sim_layout = QVBoxLayout(self.sim_tab)
        
        # Plot Widget using pyqtgraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.plot_widget = pg.GraphicsLayoutWidget()
        sim_layout.addWidget(self.plot_widget)
        
        # Time Domain Plot
        self.p_time = self.plot_widget.addPlot(title="Time Domain Signal")
        # Default; will be overwritten dynamically depending on domain (A or V)
        self.p_time.setLabel('left', 'Signal', units='A')
        self.p_time.setLabel('bottom', 'Time', units='s')
        self.p_time.addLegend()
        self.p_time.showGrid(x=True, y=True)
        # Enable Mouse Interaction for Zoom/Pan (Default in pyqtgraph, but explicit here)
        self.p_time.setMouseEnabled(x=True, y=True)
        
        self.plot_widget.nextRow()
        
        # Frequency Domain Plot
        self.p_freq = self.plot_widget.addPlot(title="Power Spectral Density (PSD)")
        # IMPORTANT: Avoid using `units=` for squared units in pyqtgraph (SI prefixes become misleading).
        # We'll render units inside the label text and switch between A^2/Hz and V^2/Hz dynamically.
        self.p_freq.setLabel('left', 'PSD (A^2/Hz)')
        self.p_freq.setLabel('bottom', 'Frequency', units='Hz')
        self.p_freq.setLogMode(x=True, y=True)
        self.p_freq.showGrid(x=True, y=True)
        self.p_freq.setMouseEnabled(x=True, y=True)
        self.p_freq.addLegend()
        
        self.plot_widget.nextRow()

        # Chebyshev Spectrum Plot
        self.p_cheb = self.plot_widget.addPlot(title="Chebyshev Noise Spectroscopy")
        # Same note as PSD: avoid `units=` with squared units; switch dynamically between A^2 and V^2.
        self.p_cheb.setLabel('left', 'Intensity (A^2)')
        self.p_cheb.setLabel('bottom', 'Spectral Line (k)')
        self.p_cheb.setLogMode(x=True, y=True)
        self.p_cheb.showGrid(x=True, y=True)
        self.p_cheb.setMouseEnabled(x=True, y=True)
        self.p_cheb.addLegend()
        
        # Layer Visibility Controls (Checkboxes below plots)
        layers_layout = QHBoxLayout()
        self.chk_show_total = QCheckBox("Total (Sampled)")
        self.chk_show_total.setChecked(True)
        self.chk_show_total.stateChanged.connect(self.update_plots)

        self.chk_show_high_res = QCheckBox("Analog (High Res)")
        self.chk_show_high_res.setChecked(True)
        self.chk_show_high_res.stateChanged.connect(self.update_plots)
        
        self.chk_show_corr = QCheckBox("Corrosion")
        self.chk_show_corr.setChecked(False)
        self.chk_show_corr.stateChanged.connect(self.update_plots)
        
        self.chk_show_bio = QCheckBox("Bio")
        self.chk_show_bio.setChecked(False)
        self.chk_show_bio.stateChanged.connect(self.update_plots)
        
        layers_layout.addWidget(QLabel("Show Layers:"))
        layers_layout.addWidget(self.chk_show_total)
        layers_layout.addWidget(self.chk_show_high_res)
        layers_layout.addWidget(self.chk_show_corr)
        layers_layout.addWidget(self.chk_show_bio)
        layers_layout.addStretch()
        
        sim_layout.addLayout(layers_layout)
        
        # Tab 2: Documentation
        self.help_tab = HelpWidget()
        self.tabs.addTab(self.help_tab, "Reference & Logic")

        # Tab 3: Coating Breach Study
        self.study_tab = StudyTab(main_window=self)
        self.tabs.addTab(self.study_tab, "Coating Breach Study")

        # Tab 4: Electrode Optimisation
        self.optimization_tab = OptimizationTab(main_window=self)
        self.tabs.addTab(self.optimization_tab, "Electrode Optimisation")

        # Hide left sidebar when study/optimisation tabs are active
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        main_layout.addLayout(right_layout)

    def _on_tab_changed(self, index):
        """Show/hide main sidebar depending on active tab."""
        w = self.tabs.widget(index)
        has_own_controls = (w is self.study_tab or w is self.optimization_tab)
        self.left_sidebar.setVisible(not has_own_controls)

    def create_global_controls(self):
        group = QGroupBox("Global Parameters")
        layout = QFormLayout()
        
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 3600.0)
        self.spin_duration.setValue(10.0)
        self.spin_duration.setSuffix(" s")
        layout.addRow("Duration:", self.spin_duration)
        
        self.spin_internal_fs = QSpinBox()
        self.spin_internal_fs.setRange(1000, 1000000)
        self.spin_internal_fs.setValue(10000)
        self.spin_internal_fs.setSuffix(" Hz")
        self.spin_internal_fs.setSingleStep(1000)
        self.spin_internal_fs.setToolTip("Frequency of physical simulation (Reality)")
        layout.addRow("Physics Rate:", self.spin_internal_fs)
        
        self.spin_fs = QSpinBox()
        self.spin_fs.setRange(1, 100000)
        self.spin_fs.setValue(1000)
        self.spin_fs.setSuffix(" Hz")
        self.spin_fs.setSingleStep(10)
        self.spin_fs.setToolTip("Frequency of measurement (ADC)")
        layout.addRow("Sampling Rate:", self.spin_fs)
        
        # ADC Type
        self.combo_adc = QComboBox()
        self.combo_adc.addItem("Ideal (Sinc5+Sinc1 equivalent)", "ideal")
        self.combo_adc.addItem("Sigma-Delta (Lowpass Decimation)", "sigma_delta")
        self.combo_adc.addItem("Naive Sampling (Aliasing!)", "naive")
        self.combo_adc.setToolTip("Simulation method for ADC downsampling")
        layout.addRow("ADC Type:", self.combo_adc)

        self.spin_adc_lsb = QDoubleSpinBox()
        self.spin_adc_lsb.setRange(0.0, 100.0)
        self.spin_adc_lsb.setValue(0.0) # 0 = infinite resolution
        self.spin_adc_lsb.setSingleStep(0.01)
        self.spin_adc_lsb.setDecimals(4)
        self.spin_adc_lsb.setSuffix(" uA")
        self.spin_adc_lsb.setToolTip("Amplitude Quantization Step (LSB). Set > 0 to simulate ADC bit depth.")
        layout.addRow("ADC Resolution (LSB):", self.spin_adc_lsb)
        
        # Enforce Internal >= Sampling
        self.spin_fs.valueChanged.connect(self._check_fs)
        self.spin_internal_fs.valueChanged.connect(self._check_fs)
        
        group.setLayout(layout)
        self.controls_layout.addWidget(group)

    def _check_fs(self):
        fs_in = self.spin_internal_fs.value()
        fs_out = self.spin_fs.value()
        if fs_in < fs_out:
            self.spin_internal_fs.setValue(fs_out)

    def create_corrosion_controls(self):
        group = QGroupBox("Corrosion & Impedance")
        layout = QFormLayout()
        
        # Source Control
        self.spin_dc = QDoubleSpinBox()
        self.spin_dc.setRange(-1000, 1000)
        self.spin_dc.setValue(50.0)
        layout.addRow("DC Current (uA):", self.spin_dc)
        
        self.combo_trend = QComboBox()
        self.combo_trend.addItem("Polynomial (Drift)", "polynomial")
        self.combo_trend.addItem("Sine (Slow Wave)", "sine")
        self.combo_trend.addItem("None (Constant)", "none")
        layout.addRow("Trend Type:", self.combo_trend)
        
        self.spin_noise_alpha = QDoubleSpinBox()
        self.spin_noise_alpha.setRange(0.0, 3.0)
        self.spin_noise_alpha.setValue(1.5)
        self.spin_noise_alpha.setSingleStep(0.1)
        layout.addRow("Noise Alpha (1/f^a):", self.spin_noise_alpha)
        
        self.spin_noise_power = QDoubleSpinBox()
        self.spin_noise_power.setRange(0.0, 100.0)
        self.spin_noise_power.setValue(1.0)
        layout.addRow("Noise Power:", self.spin_noise_power)
        
        self.spin_bubble_rate = QDoubleSpinBox()
        self.spin_bubble_rate.setRange(0.0, 100.0)
        self.spin_bubble_rate.setValue(5.0)
        layout.addRow("Bubble Rate (Hz):", self.spin_bubble_rate)
        
        self.spin_bubble_amp = QDoubleSpinBox()
        self.spin_bubble_amp.setRange(0.0, 1000.0)
        self.spin_bubble_amp.setValue(20.0)
        layout.addRow("Bubble Amp (Mean):", self.spin_bubble_amp)

        self.spin_decay = QDoubleSpinBox()
        self.spin_decay.setRange(0.001, 5.0)
        self.spin_decay.setValue(0.05)
        self.spin_decay.setSingleStep(0.01)
        layout.addRow("Bubble Decay (s):", self.spin_decay)
        
        # ── Equivalent Circuit ──────────────────────────────────────
        circuit_header = QLabel("<b>Equivalent Circuit</b>")
        layout.addRow(circuit_header)

        self.chk_use_circuit = QCheckBox("Electrode Circuit Model")
        self.chk_use_circuit.setChecked(False)
        self.chk_use_circuit.setToolTip(
            "Enable equivalent circuit physics:\n"
            "• Bio V→I conversion (V_ECG / Z_loop)\n"
            "• Johnson-Nyquist thermal noise\n"
            "• Frequency-dependent impedance filtering"
        )
        self.chk_use_circuit.stateChanged.connect(self.toggle_circuit_controls)
        layout.addRow(self.chk_use_circuit)

        self.chk_physics_mode = QCheckBox("Link Physics (Auto-calc Noise/Bubbles)")
        self.chk_physics_mode.setChecked(False)
        self.chk_physics_mode.setToolTip(
            "Derive I_corr, Bubble Rate, and Decay from Rct/Cdl (Stern-Geary / Faraday)"
        )
        self.chk_physics_mode.stateChanged.connect(self.toggle_physics_mode)
        layout.addRow(self.chk_physics_mode)

        # Circuit builder widget (preset + schematic + params)
        self.circuit_builder = CircuitBuilderWidget()
        self.circuit_builder.sigCircuitChanged.connect(self.recalc_rs)
        layout.addRow(self.circuit_builder)

        # ── Measurement Loop ──────────────────────────────────────
        meas_header = QLabel("<b>Measurement Loop</b>")
        layout.addRow(meas_header)

        self.chk_symmetric = QCheckBox("Two Identical Electrodes (Symmetric)")
        self.chk_symmetric.setChecked(False)
        self.chk_symmetric.setToolTip("Simulate difference between two independent electrodes")
        layout.addRow(self.chk_symmetric)

        self.spin_rshunt = QDoubleSpinBox()
        self.spin_rshunt.setRange(0.0, 10000.0)
        self.spin_rshunt.setValue(10.0)
        self.spin_rshunt.setSuffix(" Ω")
        self.spin_rshunt.setToolTip("Shunt resistance for current measurement")
        layout.addRow("Rshunt (ZRA):", self.spin_rshunt)

        self.spin_area = QDoubleSpinBox()
        self.spin_area.setRange(0.0001, 100.0)
        self.spin_area.setValue(0.05)
        self.spin_area.setDecimals(4)
        self.spin_area.setSuffix(" cm²")
        self.spin_area.setToolTip("Electrode area — used for specific→absolute conversion and Rs calculation")
        self.spin_area.valueChanged.connect(self.recalc_rs)
        layout.addRow("Electrode Area:", self.spin_area)

        self.spin_rho = QDoubleSpinBox()
        self.spin_rho.setRange(1.0, 10000.0)
        self.spin_rho.setValue(100.0)
        self.spin_rho.setSuffix(" Ω·cm")
        self.spin_rho.setToolTip("Tissue resistivity — used with electrode positions to calculate Rs")
        self.spin_rho.valueChanged.connect(self.recalc_rs)
        layout.addRow("Resistivity (ρ):", self.spin_rho)

        self.spin_rs = QDoubleSpinBox()
        self.spin_rs.setRange(0.1, 100000.0)
        self.spin_rs.setValue(100.0)
        self.spin_rs.setSuffix(" Ω")
        self.spin_rs.setReadOnly(True)
        self.spin_rs.setToolTip("Auto-calculated from electrode geometry and tissue resistivity")
        self.spin_rs.setStyleSheet("background-color: #f0f0f0;")
        layout.addRow("Rs (auto):", self.spin_rs)

        group.setLayout(layout)
        self.controls_layout.addWidget(group)
        self.toggle_circuit_controls()

    def toggle_circuit_controls(self):
        enabled = self.chk_use_circuit.isChecked()
        # Enable/disable the circuit builder and measurement controls
        self.circuit_builder.set_enabled_all(enabled)
        self.spin_rs.setEnabled(enabled)
        self.spin_rshunt.setEnabled(enabled)
        self.spin_rho.setEnabled(enabled)
        self.spin_area.setEnabled(True)  # area always editable (used by physics mode too)
        self.chk_symmetric.setEnabled(enabled)

        if enabled:
            self.recalc_rs()

    def toggle_physics_mode(self):
        is_physics = self.chk_physics_mode.isChecked()
        enabled_inputs = not is_physics
        
        # When physics mode is ON, these are auto-calculated
        self.spin_dc.setEnabled(enabled_inputs)
        self.spin_noise_power.setEnabled(enabled_inputs)
        self.spin_bubble_rate.setEnabled(enabled_inputs)
        self.spin_bubble_amp.setEnabled(enabled_inputs)
        self.spin_decay.setEnabled(enabled_inputs)
        
        if is_physics:
            # Apply experimental settings automatically
            self.spin_adc_lsb.setValue(0.05)   # 0.05 uA resolution
            self.spin_white_noise.setValue(0.0)# Clean steps
            self.spin_noise_alpha.setValue(2.0)# Brownian walk
            
            # Physics requires impedance parameters, so force circuit ON
            if not self.chk_use_circuit.isChecked():
                self.chk_use_circuit.setChecked(True)
            self.chk_use_circuit.setEnabled(False) # Lock it
        else:
            self.chk_use_circuit.setEnabled(True)

    def recalc_rs(self):
        if not self.chk_use_circuit.isChecked():
            return

        # Get positions
        pos = self.electrode_widget.get_positions()
        w = np.array(pos['working'])
        c = np.array(pos['counter'])

        # Distance on plot (0..1 units)
        dist_plot = np.linalg.norm(w - c)

        # Convert to physical cm
        # Body length in plot units is approx 0.6 (from 0.2 to 0.8 y)
        # Body length in reality is self.electrode_widget.MOUSE_SIZE_CM (5.0 cm)
        scale = self.electrode_widget.MOUSE_SIZE_CM / 0.6
        dist_cm = dist_plot * scale

        # Rs = R_contact + rho * dist_cm / A_eff
        # Use electrode area from the spin box
        A_eff = max(self.spin_area.value(), 0.001)
        rho = self.spin_rho.value()
        r_contact = 10.0

        rs_val = r_contact + (rho * dist_cm / A_eff)
        self.spin_rs.setValue(rs_val)

    def create_bio_controls(self):
        group = QGroupBox("Bio-Signals Engine")
        layout = QFormLayout()
        
        # ECG
        self.chk_ecg = QCheckBox("Enable ECG")
        self.chk_ecg.setChecked(True)
        layout.addRow(self.chk_ecg)
        
        self.spin_hr = QDoubleSpinBox()
        self.spin_hr.setRange(60, 1000)
        self.spin_hr.setValue(400)
        layout.addRow("Heart Rate (bpm):", self.spin_hr)
        
        self.spin_ecg_amp = QDoubleSpinBox()
        # Interpreted as induced differential voltage (mV) between electrodes
        self.spin_ecg_amp.setRange(0.0, 10.0)
        # Increased default to 1.5 mV to represent stronger near-field potential in mouse model
        self.spin_ecg_amp.setValue(1.5)
        self.spin_ecg_amp.setSuffix(" mV")
        layout.addRow("Ref. ECG Amp (Lead II):", self.spin_ecg_amp)
        
        # EMG
        self.chk_emg = QCheckBox("Enable EMG")
        self.chk_emg.setChecked(False)
        layout.addRow(self.chk_emg)
        
        self.spin_emg_int = QDoubleSpinBox()
        # Interpreted as induced differential voltage (mV) between electrodes
        self.spin_emg_int.setRange(0.0, 10.0)
        self.spin_emg_int.setValue(0.05)
        self.spin_emg_int.setSuffix(" mV")
        layout.addRow("EMG Intensity (Vdiff):", self.spin_emg_int)
        
        self.spin_emg_prob = QDoubleSpinBox()
        self.spin_emg_prob.setRange(0.0, 1.0)
        self.spin_emg_prob.setValue(0.1)
        self.spin_emg_prob.setSingleStep(0.05)
        layout.addRow("Burst Probability:", self.spin_emg_prob)
        
        group.setLayout(layout)
        self.controls_layout.addWidget(group)

    def create_sensor_controls(self):
        group = QGroupBox("Sensor/Artifacts")
        layout = QFormLayout()
        
        self.spin_white_noise = QDoubleSpinBox()
        self.spin_white_noise.setRange(0.0, 100.0)
        self.spin_white_noise.setValue(0.2)
        layout.addRow("White Noise Level:", self.spin_white_noise)
        
        self.spin_mains = QDoubleSpinBox()
        self.spin_mains.setRange(0.0, 100.0)
        self.spin_mains.setValue(0.0)
        layout.addRow("50Hz Hum Level:", self.spin_mains)
        
        group.setLayout(layout)
        self.controls_layout.addWidget(group)

    def get_params(self):
        # Units policy:
        # - All currents inside engines are in Amps (A).
        # - UI current-related knobs are expressed in microamps (uA), so we convert by 1e-6 here.
        # NOTE: Even when the impedance model is enabled, the primary measurement in this app is CURRENT (ZRA/shunt).
        use_circuit = self.chk_use_circuit.isChecked()
        current_scale = 1e-6
        sensor_scale = 1e-6  # SensorEngine is interpreted as current noise/artifacts (A) in all modes
        bio_v_scale = 1e-3   # mV -> V (bio branch is induced differential voltage)

        # Build composable circuit from the circuit builder
        area = self.spin_area.value()
        electrode_circuit = None
        R_ct_value = None
        C_dl_effective = None

        if use_circuit or self.chk_physics_mode.isChecked():
            try:
                circuit, rct_abs, cdl_eff = self.circuit_builder.get_circuit(area)
                electrode_circuit = circuit
                R_ct_value = rct_abs
                C_dl_effective = cdl_eff
            except Exception as e:
                print(f"Circuit build error: {e}")

        return {
            "duration": self.spin_duration.value(),
            "fs": self.spin_fs.value(),
            "internal_fs": self.spin_internal_fs.value(),
            "adc_type": self.combo_adc.currentData(),
            "adc_lsb": self.spin_adc_lsb.value() * 1e-6,  # uA -> A
            "electrodes": self.electrode_widget.get_positions(),
            "corrosion": {
                "dc_current": self.spin_dc.value() * current_scale,
                "trend_type": self.combo_trend.currentData(),
                "noise_alpha": self.spin_noise_alpha.value(),
                "noise_power": self.spin_noise_power.value() * current_scale,
                "bubble_rate": self.spin_bubble_rate.value(),
                "bubble_amp_mean": self.spin_bubble_amp.value() * current_scale,
                "bubble_decay": self.spin_decay.value(),
                "physics_aware_mode": self.chk_physics_mode.isChecked(),
                # Equivalent Circuit
                "use_circuit_model": use_circuit,
                "use_symmetric_electrodes": self.chk_symmetric.isChecked(),
                "R_s": self.spin_rs.value(),
                "R_ct": R_ct_value if R_ct_value else 1000.0,
                "C_dl": C_dl_effective if C_dl_effective else 10.0e-6,
                "R_shunt": self.spin_rshunt.value(),
                # Composable circuit
                "electrode_circuit": electrode_circuit,
                "R_ct_value": R_ct_value,
                "C_dl_effective": C_dl_effective,
            },
            "bio": {
                "ecg_enabled": self.chk_ecg.isChecked(),
                "ecg_rate": self.spin_hr.value(),
                "ecg_amp": self.spin_ecg_amp.value() * bio_v_scale,
                "emg_enabled": self.chk_emg.isChecked(),
                "emg_intensity": self.spin_emg_int.value() * bio_v_scale,
                "emg_burst_prob": self.spin_emg_prob.value()
            },
            "sensor": {
                "white_noise_level": self.spin_white_noise.value() * sensor_scale,
                "mains_hum_level": self.spin_mains.value() * sensor_scale
            }
        }

    def _apply_domain_labels(self):
        """
        Keep plot labels consistent with the current signal domain.
        """
        # For readability we display CURRENT in microamps (uA) by default and plot PSD/Chebyshev in uA^2.
        # This avoids confusing SI-prefix behavior on squared units in pyqtgraph.
        if self.current_domain == "V":
            self._display_y_scale = 1e3  # V -> mV  (kept for future OCP mode)
            self.p_time.setLabel('left', 'Voltage (mV)')
            self.p_freq.setLabel('left', 'PSD (mV²/Hz)')
            self.p_cheb.setLabel('left', 'Intensity (mV²)')
        else:
            self._display_y_scale = 1e6  # A -> uA
            self.p_time.setLabel('left', 'Current', units='uA')
            self.p_freq.setLabel('left', 'PSD (uA^2/Hz)')
            self.p_cheb.setLabel('left', 'Intensity (uA^2)')

        # Keep squared axes from auto-prefixing (we already control scale)
        try:
            self.p_freq.getAxis('left').enableAutoSIPrefix(False)
            self.p_cheb.getAxis('left').enableAutoSIPrefix(False)
        except Exception:
            pass

    def generate_signal(self):
        params = self.get_params()
        try:
            self.current_data = self.compositor.generate(params)
            # Output is always ZRA shunt current (Amps), regardless of
            # whether the impedance model is active or not.
            self.current_domain = "A"
            self._apply_domain_labels()
            self.update_plots()
        except Exception as e:
            print(f"Generation Error: {e}")
            import traceback
            traceback.print_exc()

    def update_plots(self):
        if self.current_data is None:
            return
            
        # Ensure labels & display scaling are consistent every redraw
        self._apply_domain_labels()
        y_scale = getattr(self, "_display_y_scale", 1.0)

        t = self.current_data["time"]
        
        self.p_time.clear()
        self.p_time.addLegend()
        self.p_cheb.clear()
        
        # Determine what to plot
        # 1. High Res "Analog" Signal
        if self.chk_show_high_res.isChecked() and "signal_high_res" in self.current_data:
             t_hr = self.current_data["time_high_res"]
             sig_hr = self.current_data["signal_high_res"] * y_scale
             # Downsample for visualization if too large (> 100k points) to keep UI responsive
             if len(sig_hr) > 100000:
                 factor = len(sig_hr) // 100000
                 t_hr = t_hr[::factor]
                 sig_hr = sig_hr[::factor]
                 
             self.p_time.plot(t_hr, sig_hr, pen=pg.mkPen(color=(200, 200, 200), width=2), name='Analog (High Res)')

        # 2. Sampled Signal
        if self.chk_show_total.isChecked():
            # Use symbol to show sampling points if not too dense
            symbol = 'o' if len(t) < 500 else None
            self.p_time.plot(t, self.current_data["total_signal"] * y_scale, pen=pg.mkPen('k', width=2), symbol=symbol, symbolSize=5, symbolBrush='k', name='Sampled')
            
        if self.chk_show_corr.isChecked():
            # Plot High Res Corrosion if available, otherwise Sampled
            if "corrosion_high_res" in self.current_data and self.chk_show_high_res.isChecked():
                t_hr = self.current_data["time_high_res"]
                sig_hr = self.current_data["corrosion_high_res"] * y_scale
                # Downsample for visualization
                if len(sig_hr) > 100000:
                    factor = len(sig_hr) // 100000
                    t_hr = t_hr[::factor]
                    sig_hr = sig_hr[::factor]
                self.p_time.plot(t_hr, sig_hr, pen=pg.mkPen(color=(255, 100, 100, 200), width=2), name='Corrosion (HR)')
            else:
                self.p_time.plot(t, self.current_data["corrosion_component"] * y_scale, pen=pg.mkPen('r', width=2), name='Corrosion')
            
        if self.chk_show_bio.isChecked():
            # Sum of ECG + EMG if enabled
            # Logic: If "Analog (High Res)" is ON, show the Source (High Res) Bio.
            # If OFF, show the Sampled Bio (what is captured).
            
            if "bio_high_res" in self.current_data and self.chk_show_high_res.isChecked():
                t_hr = self.current_data["time_high_res"]
                sig_hr = self.current_data["bio_high_res"] * y_scale
                 # Downsample for visualization
                if len(sig_hr) > 100000:
                    factor = len(sig_hr) // 100000
                    t_hr = t_hr[::factor]
                    sig_hr = sig_hr[::factor]
                self.p_time.plot(t_hr, sig_hr, pen=pg.mkPen(color=(100, 255, 100, 200), width=2), name='Bio (Source)')
            else:
                bio_sum = self.current_data["bio_component"]
                self.p_time.plot(t, bio_sum * y_scale, pen=pg.mkPen('g', width=2), name='Bio (Sampled)')

        # Update PSD
        self.update_psd()
        
        # Update Chebyshev Spectrum
        self.update_chebyshev()

    def get_chebyshev_basis(self, N):
        """
        Constructs orthonormal DISCRETE Chebyshev polynomials (Gram polynomials) on x=0..N-1.

        We reproduce the classic definition used in "Chebyshev noise spectroscopy" papers:

        Discrete Chebyshev polynomials t_n(x) satisfy (for fixed N):
            t_0(x) = 1
            t_1(x) = 2x - N + 1
            (n+1) t_{n+1}(x) = (2n+1)(2x - N + 1) t_n(x) - n (N^2 - n^2) t_{n-1}(x)

        with orthogonality over x=0..N-1 (weight 1).

        Orthonormal basis is:
            C[n, x] = t_n(x) / H_n
        where
            H_n^2 = N * Π_{k=1..n} (N^2 - k^2) / (2n+1)

        Returns:
            C: shape (N, N), rows are basis vectors for projection Y = C @ u
        """
        if hasattr(self, "_cheb_basis") and self._cheb_basis.shape == (N, N):
            return self._cheb_basis

        x = np.arange(N, dtype=float)

        # Evaluate discrete Chebyshev polynomials t_n(x) for n=0..N-1
        tnx = np.zeros((N, N), dtype=float)  # (n, x)
        tnx[0, :] = 1.0
        if N > 1:
            tnx[1, :] = 2.0 * x - (N - 1)
        for n in range(1, N - 1):
            tnx[n + 1, :] = (
                (2 * n + 1) * (2.0 * x - (N - 1)) * tnx[n, :]
                - n * (N**2 - n**2) * tnx[n - 1, :]
            ) / (n + 1)

        # Norms H_n computed in log-space for numerical stability
        log_h2 = np.zeros(N, dtype=float)
        for n in range(N):
            if n == 0:
                log_h2[n] = np.log(N) - np.log(1.0)  # N/(2*0+1)
            else:
                ks = np.arange(1, n + 1, dtype=float)
                log_h2[n] = np.log(N) + np.sum(np.log(N**2 - ks**2)) - np.log(2 * n + 1)
        h = np.exp(0.5 * log_h2)

        C = tnx / h[:, None]

        # Cache
        self._cheb_basis = C
        return self._cheb_basis

    def update_chebyshev(self):
        if self.current_data is None:
            return
            
        # Get signal
        y = self.current_data["total_signal"]
        y_scale = getattr(self, "_display_y_scale", 1.0)
        intensity_scale = y_scale ** 2
        
        # Algorithm parameters from paper
        N = 16 
        
        if len(y) < N:
            return

        # 1. Split into segments
        M = len(y) // N
        if M < 1: return
        
        segments = y[:M*N].reshape(M, N)
        
        # 2. Get Basis
        try:
            basis = self.get_chebyshev_basis(N) # (N, N)
            
            # 3. Transform: Y = C * u (for each segment)
            # segments is (M, N). We transpose to (N, M) for matmul
            coeffs = basis @ segments.T # Result: (N, M)
            
            # 4. Intensity = Square
            intensities = (coeffs**2) * intensity_scale
            
            # 5. Median Average
            median_spectrum = np.median(intensities, axis=1)
            
            # 6. Plot
            # Plot lines k=2..15 (skip DC and Trend as per paper)
            k_indices = np.arange(2, N)
            spec_data = median_spectrum[2:]
            
            # Remove zeros for log plot
            spec_data[spec_data <= 0] = 1e-20
            
            self.p_cheb.clear()
            try:
                self.p_cheb.getAxis('left').enableAutoSIPrefix(False)
            except Exception:
                pass
            self.p_cheb.plot(k_indices, spec_data, pen=pg.mkPen('b', width=3), symbol='o', name="Median Spectrum")
            
            # Optional: Show some individual segments (faint) to show dispersion
            # Limit to 50 segments max to avoid lag
            limit = min(M, 50)
            step = M // limit
            for i in range(0, M, step):
                seg_spec = intensities[2:, i]
                seg_spec[seg_spec <= 0] = 1e-20
                self.p_cheb.plot(k_indices, seg_spec, pen=pg.mkPen((100, 100, 100, 50), width=1))
                
        except Exception as e:
            print(f"Chebyshev analysis error: {e}")

    def update_psd(self):
        if self.current_data is None:
            return
            
        self.p_freq.clear()
        # Re-add legend if cleared (pyqtgraph clear() removes items, legend persists usually but items need name)
        
        fs = self.spin_fs.value()

        # Use the current display scaling (A->uA implies PSD scale of (1e6)^2 = 1e12)
        y_scale = getattr(self, "_display_y_scale", 1.0)
        psd_scale = y_scale ** 2
        
        # 1. Total Signal
        if self.chk_show_total.isChecked():
            sig = self.current_data["total_signal"]
            f, Pxx = signal.welch(sig, fs, nperseg=min(len(sig), 4096))
            self.p_freq.plot(f, Pxx * psd_scale, pen=pg.mkPen('k', width=2), name='Total')

        # 2. Corrosion Component
        if self.chk_show_corr.isChecked():
            sig_corr = self.current_data["corrosion_component"]
            f, Pxx = signal.welch(sig_corr, fs, nperseg=min(len(sig_corr), 4096))
            self.p_freq.plot(f, Pxx * psd_scale, pen=pg.mkPen('r', width=2), name='Corrosion')
            
        # 3. Bio Component
        if self.chk_show_bio.isChecked():
            sig_bio = self.current_data["bio_component"]
            f, Pxx = signal.welch(sig_bio, fs, nperseg=min(len(sig_bio), 4096))
            self.p_freq.plot(f, Pxx * psd_scale, pen=pg.mkPen('g', width=2), name='Bio')
            
        # Optional: Sensor noise? 
        # Only if explicitly wanted or maybe just leave it as part of total. 
        # User asked for "constituents", usually Sensor is one. 
        # But we don't have a checkbox for "Sensor" in layers, only Corr and Bio.
        # Let's stick to what we have controls for to avoid clutter, 
        # or maybe add it if Total is checked but very faint? No, let's keep it clean.

    def load_experimental_csv(self):
        """
        Load an experimental CSV and display it with PSD and Chebyshev noise spectroscopy.

        Expected input format (user-provided):
        df = pd.read_csv(file, skiprows=1, encoding='latin1').reset_index()
        df.rename(columns={'index': 'time_0', 'Channel 0': 'current'}, inplace=True)

        Current conversion:
        df['current'] = ((df['current'] - 2**23) * 5.69 * 1e-2)
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Experimental CSV",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if not path:
            return

        default_fs = int(self.spin_fs.value()) if int(self.spin_fs.value()) > 0 else 10
        fs, ok = QInputDialog.getInt(
            self,
            "Sampling Rate",
            "Sampling rate (Hz):",
            value=default_fs,
            min=1,
            max=100000,
            step=1
        )
        if not ok:
            return

        try:
            df = pd.read_csv(path, skiprows=1, encoding="latin1").reset_index()

            if "Channel 0" not in df.columns:
                raise ValueError(f"CSV must contain 'Channel 0' column. Found: {list(df.columns)}")

            df.rename(columns={"index": "time_0", "Channel 0": "current"}, inplace=True)

            # Convert ADC code to current.
            # User equation: ((code - 2**23) * 5.69 * 1e-2)
            # In many pipelines, 5.69e-2 is in microamps per LSB; to express current in Amps we multiply by 1e-6.
            # This makes Chebyshev intensities land in the expected ~1e-12 A^2 range for ~uA-level signals.
            df["current"] = (df["current"].astype(float) - 2**23) * (5.69e-2)*1e-6

            current = df["current"].to_numpy(dtype=float)
            time = np.arange(len(current), dtype=float) / float(fs)

            # Update sampling rate control so PSD uses correct fs
            self.spin_fs.setValue(int(fs))
            self.current_domain = "A"
            self._apply_domain_labels()

            # Keep y-axis SI prefix on linear axis (A), but disable prefixing on squared axes.
            try:
                self.p_cheb.getAxis('left').enableAutoSIPrefix(False)
                self.p_freq.getAxis('left').enableAutoSIPrefix(False)
            except Exception:
                pass

            zeros = np.zeros_like(current)
            self.current_data = {
                "time": time,
                "total_signal": current,
                "corrosion_component": zeros,
                "bio_component": zeros,
                "sensor_component": zeros,
                # Provide "high-res" equivalents for consistent plotting toggles
                "time_high_res": time,
                "signal_high_res": current,
                "corrosion_high_res": zeros,
                "bio_high_res": zeros,
                "sensor_high_res": zeros,
                # Metadata
                "source": np.array(["experimental"] * len(current), dtype=object),
            }

            # Switch to simulation tab and update plots
            if hasattr(self, "tabs"):
                self.tabs.setCurrentIndex(0)

            self.update_plots()

            # Sanity check: variance vs integral of PSD (Parseval for Welch estimate).
            # Welch detrends by default (constant), so compare to variance of (x - mean).
            try:
                x = current - np.mean(current)
                nperseg = min(len(x), 4096)
                f, Pxx = signal.welch(x, fs, nperseg=nperseg)
                var_time = float(np.var(x))
                var_psd = float(np.trapz(Pxx, f))

                # Show in microamps for readability
                rms_uA = float(np.std(x) * 1e6)
                peak_uA = float(np.max(np.abs(x)) * 1e6)

                QMessageBox.information(
                    self,
                    "Experimental sanity-check",
                    f"fs = {fs} Hz, n = {len(x)}, nperseg = {nperseg}\n"
                    f"RMS = {rms_uA:.3f} µA, peak = {peak_uA:.3f} µA\n"
                    f"var(time) = {var_time:.3e} A^2\n"
                    f"∫PSD df ≈ {var_psd:.3e} A^2\n"
                    f"(These should be the same order of magnitude.)"
                )
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "CSV Load Error", str(e))

    def export_data(self):
        if self.current_data is None:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;NumPy (*.npy)")
        if not path:
            return
            
        df = self.compositor.to_dataframe(self.current_data)
        
        if path.endswith('.csv'):
            df.to_csv(path, index=False)
        elif path.endswith('.npy'):
            np.save(path, self.current_data)
        
        print(f"Saved to {path}")

    def reset_zoom(self):
        self.p_time.autoRange()
        self.p_freq.autoRange()
        if hasattr(self, "p_cheb"):
            self.p_cheb.autoRange()