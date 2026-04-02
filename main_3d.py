import sys
import os
import numpy as np
from PyQt6.QtWidgets import QApplication, QSplitter, QWidget, QHBoxLayout, QScrollArea
from PyQt6.QtCore import Qt
from src.ui import MainWindow
from src.widget_3d import BioPhantomViewer

# Force Qt to use OpenGL (fix for D3D11 black screen on Windows)
os.environ["QSG_RHI_BACKEND"] = "opengl"

class MainWindow3D(MainWindow):
    """
    Extends the original MainWindow but injects the 3D widget 
    in place of the 2D schematic, and adds a Splitter for resizing.
    """
    def init_ui(self):
        # Call parent to build the standard UI
        super().init_ui()
        
        # --- 1. Swap 2D widget for 3D ---
        self.controls_layout.removeWidget(self.electrode_widget)
        self.electrode_widget.deleteLater()
        
        self.electrode_widget = BioPhantomViewer()
        self.electrode_widget.setMinimumHeight(500)
        self.electrode_widget.geometry_changed.connect(self.on_3d_geometry_changed)
        
        # Insert at correct position
        self.controls_layout.insertWidget(4, self.electrode_widget)
        
        # --- 2. Add Resizable Splitter ---
        central = self.centralWidget()
        main_layout = central.layout()
        
        scroll_item = main_layout.itemAt(0)
        scroll_widget = scroll_item.widget()
        
        if isinstance(scroll_widget, QScrollArea):
            scroll_widget.setFixedWidth(16777215) 
            scroll_widget.setMinimumWidth(350) 
        
        tabs_widget = self.tabs
        
        scroll_widget.setParent(None) 
        tabs_widget.setParent(None)   
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll_widget)
        splitter.addWidget(tabs_widget)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        while main_layout.count():
            item = main_layout.takeAt(0)
        
        main_layout.addWidget(splitter)

    def on_3d_geometry_changed(self, gain, pos_w, pos_c):
        self._cached_lead_gain = gain
        # Trigger recalculation of Rs
        self.recalc_rs()

    # Override the parent's recalc_rs to use 3D logic
    def recalc_rs(self):
        if not self.chk_use_circuit.isChecked():
            return
            
        # Get positions from internal phantom state (cm)
        # Accessing private props of the 3D widget
        p_w = self.electrode_widget.pos_w_local
        p_c = self.electrode_widget.pos_c_local
        
        # Distance in cm directly from 3D model
        dist_cm = np.linalg.norm(p_w - p_c)
        
        rho = self.spin_rho.value()
        
        # Sync physics with 3D phantom
        self.electrode_widget.set_tissue_resistivity(rho)
        
        # Assume electrode area ~ 0.5 cm^2 (sphere r=0.35 -> A=4*pi*r^2 approx 1.5, half contact ~0.7)
        A_eff = 0.5
        r_contact = 10.0
        
        rs_val = r_contact + (rho * dist_cm / A_eff)
        
        # Block signals to prevent recursion if needed, though setValue usually safe
        self.spin_rs.blockSignals(True)
        self.spin_rs.setValue(rs_val)
        self.spin_rs.blockSignals(False)

    def get_params(self):
        params = super().get_params()
        # Inject the calculated gain from 3D model
        # (skip compositor's 2D dipole model — the 3D phantom already
        #  computes the geometric gain from the volume-conductor solution)
        if hasattr(self, '_cached_lead_gain'):
            params['bio']['lead_gain'] = self._cached_lead_gain
            params['bio']['skip_dipole_model'] = True
        return params

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow3D()
    window.show()
    sys.exit(app.exec())
