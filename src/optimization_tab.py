"""
Electrode Optimisation Study — interactive GUI tab.

Monte Carlo sweep over electrode area and coating resistance to produce:
  (a) Bare-metal reference plot (corrosion RMS vs ECG vs floor)
  (b) SNR phase diagram  (log10(SNR) vs Rf, A)

Uses the same two-zone impedance model and physics engine as the
Coating Breach Study tab.
"""

import os
import time
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom as ndimage_zoom

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QScrollArea,
    QTabWidget, QTextEdit, QComboBox, QProgressBar,
    QFileDialog, QSplitter, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from .physics_engine import CorrosionEngine, BioEngine
from .circuit_model import (
    build_electrode_circuit, convert_to_absolute,
    PRESETS, get_active_param_names, Parallel,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build mixed circuit (same logic as study_tab._build_mixed_circuit)
# ═══════════════════════════════════════════════════════════════════════════

def _build_mixed_circuit(f, area, R_f_sp,
                         coated_specific, coated_topo,
                         rho_tissue, electrode_dist, r_contact):
    """Build two-zone circuit for a given breach fraction and Rf override."""
    R_s = r_contact + (rho_tissue * electrode_dist / max(area, 1e-4))

    interface_keys = {"Rct", "Y0_dl", "n_dl"}
    if coated_topo.get("has_warburg"):
        interface_keys.add("sigma_w")
    if coated_topo.get("has_inductive"):
        interface_keys |= {"L", "RL"}

    interface_specific = {k: v for k, v in coated_specific.items()
                          if k in interface_keys}

    film_specific = {
        "Rf":   R_f_sp,
        "Y0_f": coated_specific.get("Y0_f", 800.0),
        "n_f":  coated_specific.get("n_f", 0.70),
    }

    has_warburg = coated_topo.get("has_warburg", False)
    has_inductive = coated_topo.get("has_inductive", False)

    Rct_sp = interface_specific.get("Rct", 300.0)

    if f >= 1.0:
        # Pure bare metal — no film
        ab = convert_to_absolute(interface_specific, area)
        circuit, R_ct, C_dl = build_electrode_circuit(
            ab, has_film=False,
            has_warburg=has_warburg, has_inductive=has_inductive)
        R_p = Rct_sp / area
        return _MCResult(circuit, circuit, circuit, None,
                         R_p, C_dl, R_p, C_dl, R_s)

    if f <= 0.0:
        # Pure coated — R_p includes film resistance (Stern-Geary)
        full_sp = dict(interface_specific)
        full_sp.update(film_specific)
        ab = convert_to_absolute(full_sp, area)
        circuit, R_ct, C_dl = build_electrode_circuit(
            ab, has_film=True,
            has_warburg=has_warburg, has_inductive=has_inductive)
        R_p = (Rct_sp + R_f_sp) / area
        return _MCResult(circuit, circuit, circuit, None,
                         R_p, C_dl, R_p, C_dl, R_s)

    area_bare = f * area
    area_coated = (1.0 - f) * area

    # Bare zone — interface only
    ab_b = convert_to_absolute(interface_specific, area_bare)
    circuit_bare, _, Cdl_bare = build_electrode_circuit(
        ab_b, has_film=False,
        has_warburg=has_warburg, has_inductive=has_inductive)
    R_p_bare = Rct_sp / area_bare

    # Coated zone — film + interface; R_p includes Rf (Stern-Geary)
    full_sp = dict(interface_specific)
    full_sp.update(film_specific)
    ab_c = convert_to_absolute(full_sp, area_coated)
    circuit_coated, _, Cdl_coated = build_electrode_circuit(
        ab_c, has_film=True,
        has_warburg=has_warburg, has_inductive=has_inductive)
    R_p_coated = (Rct_sp + R_f_sp) / area_coated

    # Z_ct of coated zone (interface only, no film)
    ab_ct_c = convert_to_absolute(interface_specific, area_coated)
    Z_ct_coated_circ, _, _ = build_electrode_circuit(
        ab_ct_c, has_film=False,
        has_warburg=has_warburg, has_inductive=has_inductive)

    circuit = Parallel(circuit_coated, circuit_bare)

    return _MCResult(circuit, circuit_bare, circuit_coated,
                     Z_ct_coated_circ,
                     R_p_bare, Cdl_bare, R_p_coated, Cdl_coated, R_s)


class _MCResult:
    __slots__ = ('circuit', 'circuit_bare', 'circuit_coated',
                 'Z_ct_coated',
                 'R_p_bare', 'C_dl_bare', 'R_p_coated', 'C_dl_coated',
                 'R_s')

    def __init__(self, circuit, circuit_bare, circuit_coated,
                 Z_ct_coated,
                 R_p_bare, C_dl_bare, R_p_coated, C_dl_coated, R_s):
        self.circuit = circuit
        self.circuit_bare = circuit_bare
        self.circuit_coated = circuit_coated
        self.Z_ct_coated = Z_ct_coated
        self.R_p_bare = R_p_bare
        self.C_dl_bare = C_dl_bare
        self.R_p_coated = R_p_coated
        self.C_dl_coated = C_dl_coated
        self.R_s = R_s


# ═══════════════════════════════════════════════════════════════════════════
# Core MC sweep
# ═══════════════════════════════════════════════════════════════════════════

_OPT_CFG = {}


def _run_area_sweep(f, R_f_sp, A_grid, N_real, V_fft, omega, n_hr,
                    cfg, r_shunt, seed_offset=0):
    rms_corr = np.zeros((len(A_grid), N_real))
    rms_bio = np.zeros(len(A_grid))

    for ia, A in enumerate(A_grid):
        mc = _build_mixed_circuit(
            f, A, R_f_sp,
            cfg["coated_specific"], cfg["coated_topo"],
            cfg["rho_tissue"], cfg["electrode_dist"], cfg["r_contact"])

        Z_b = mc.circuit_bare.impedance(omega)
        Z_c = mc.circuit_coated.impedance(omega)
        Z_elec = mc.circuit.impedance(omega)

        R_ext = mc.R_s + Z_elec + r_shunt

        if mc.Z_ct_coated is not None:
            Z_ct_c = mc.Z_ct_coated.impedance(omega)
            denom_b = Z_b * Z_c + R_ext * (Z_b + Z_c)
            safe_db = np.where(np.abs(denom_b) > 1e-20, denom_b, 1e10 + 0j)
            H_bare = Z_b * Z_c / safe_db

            Z_film = Z_c - Z_ct_c
            safe_br = np.where(np.abs(Z_b + R_ext) > 1e-20,
                               Z_b + R_ext, 1e10 + 0j)
            Z_b_ext = Z_b * R_ext / safe_br
            denom_c = Z_ct_c + Z_film + Z_b_ext
            safe_dc = np.where(np.abs(denom_c) > 1e-20, denom_c, 1e10 + 0j)
            H_coated = (Z_ct_c / safe_dc) * (Z_b / safe_br)
        else:
            Z_loop_circ = mc.R_s + 2.0 * Z_elec + r_shunt
            safe_zl2 = np.where(np.abs(Z_loop_circ) > 1e-10,
                                Z_loop_circ, 1e10 + 0j)
            H_bare = Z_elec / safe_zl2
            H_coated = np.zeros_like(H_bare)

        Z_loop = mc.R_s + 2.0 * Z_elec + r_shunt
        safe_zl = np.where(np.abs(Z_loop) > 1e-10, Z_loop, 1e10 + 0j)
        i_bio = np.fft.irfft(V_fft / safe_zl, n=n_hr)
        rms_bio[ia] = float(np.std(i_bio))

        _common = dict(
            use_circuit_model=True,
            use_symmetric_electrodes=True,
            physics_aware_mode=True,
            R_s=mc.R_s, R_shunt=r_shunt,
            trend_type="polynomial",
        )
        engine_bare = CorrosionEngine(
            **_common,
            electrode_circuit=mc.circuit_bare,
            R_ct_value=mc.R_p_bare,
            C_dl_effective=mc.C_dl_bare,
        )
        engine_coated = CorrosionEngine(
            **_common,
            electrode_circuit=mc.circuit_coated,
            R_ct_value=mc.R_p_coated,
            C_dl_effective=mc.C_dl_coated,
        )

        for j in range(N_real):
            np.random.seed(seed_offset + ia * 1000 + j)
            warnings.filterwarnings("ignore")

            i_bare_raw, _ = engine_bare.generate_current(
                cfg["duration"], cfg["internal_fs"])
            i_coated_raw, _ = engine_coated.generate_current(
                cfg["duration"], cfg["internal_fs"])

            I_fft = (np.fft.rfft(i_bare_raw) * H_bare
                     + np.fft.rfft(i_coated_raw) * H_coated)
            i_corr = np.fft.irfft(I_fft, n=n_hr)
            rms_corr[ia, j] = float(np.std(i_corr))

    return rms_corr, rms_bio


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _make_optimization_figure(rc_bare, re_bare, med_bare,
                              snr_map, A_detail, A_sweep, RF_grid,
                              amplifier_noise):
    uA = 1e6

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5))

    # Panel (a): bare metal
    for j in range(rc_bare.shape[1]):
        ax1.loglog(A_detail, rc_bare[:, j] * uA, 'o',
                   color='steelblue', alpha=0.10, ms=2.0,
                   zorder=3, rasterized=True)

    p10, p90 = np.percentile(rc_bare, [10, 90], axis=1)
    ax1.fill_between(A_detail, p10 * uA, p90 * uA,
                     color='steelblue', alpha=0.20, zorder=2)
    ax1.loglog(A_detail, med_bare * uA, '-',
               color='midnightblue', lw=2.5,
               label='Corrosion noise (MC median)', zorder=5)

    ax1.loglog(A_detail, re_bare * uA, '-',
               color='forestgreen', lw=2.5,
               label='ECG interference', zorder=5)
    ax1.axhline(amplifier_noise * uA, color='crimson', ls=':',
                lw=2.0, label=f'Amplifier floor ({amplifier_noise*uA:.2f} \u00b5A)')

    ax1.set_xlabel('Electrode area A (cm\u00b2)')
    ax1.set_ylabel('Current RMS (\u00b5A)')
    ax1.set_xlim(A_detail[0], A_detail[-1])
    ax1.set_ylim(1e-3, 1e1)
    ax1.grid(True, which='both', ls='-', alpha=0.12)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8)

    # Panel (b): SNR phase diagram
    log_snr = np.log10(np.clip(snr_map, 1e-2, 1e2))
    zoom_factor = 5
    log_snr_smooth = ndimage_zoom(log_snr, zoom_factor, order=3)
    nA_f = log_snr_smooth.shape[0]
    nR_f = log_snr_smooth.shape[1]
    RF_fine = np.logspace(np.log10(RF_grid[0]), np.log10(RF_grid[-1]), nR_f)
    A_fine = np.logspace(np.log10(A_sweep[0]), np.log10(A_sweep[-1]), nA_f)

    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    pcm = ax2.pcolormesh(
        RF_fine, A_fine, log_snr_smooth,
        cmap='RdYlGn', norm=norm,
        shading='nearest', rasterized=True)

    ax2.set_xlim(RF_grid[0], RF_grid[-1])
    ax2.set_ylim(A_sweep[0], A_sweep[-1])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Coating resistance Rf\u02e2\u1d56 (\u03a9\u00b7cm\u00b2)')
    ax2.set_ylabel('Electrode area A (cm\u00b2)')

    cbar = fig.colorbar(pcm, ax=ax2,
                        label='log\u2081\u2080(SNR)',
                        shrink=0.88, pad=0.02, aspect=25)
    cbar.ax.axhline(0, color='k', lw=1.5)

    ax2.text(0.40, 0.82, 'Corrosion\ndominates',
             transform=ax2.transAxes,
             fontsize=11, fontweight='bold', ha='center',
             color='#004d00',
             bbox=dict(fc='white', ec='none', alpha=0.7, pad=3))
    ax2.text(0.72, 0.18, 'ECG /\nfloor',
             transform=ax2.transAxes,
             fontsize=11, fontweight='bold', ha='center',
             color='#8B0000',
             bbox=dict(fc='white', ec='none', alpha=0.7, pad=3))

    ax1.set_title('(a) Bare metal: signal vs. interference',
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_title('(b) SNR phase diagram: coating resistance vs. area',
                  fontsize=11, fontweight='bold', pad=8)

    fig.tight_layout(w_pad=3)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Worker (QThread)
# ═══════════════════════════════════════════════════════════════════════════

class OptimizationWorker(QThread):
    sig_progress = pyqtSignal(int, int)
    sig_status = pyqtSignal(str)
    sig_finished = pyqtSignal(object)  # dict with all result arrays

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        cfg = self.cfg
        t0 = time.time()
        duration = cfg["duration"]
        internal_fs = cfg["internal_fs"]
        n_hr = int(duration * internal_fs)
        r_shunt = cfg["r_shunt"]
        amplifier_noise = cfg["amplifier_noise"]

        A_detail = np.logspace(
            np.log10(cfg["a_min"]), np.log10(cfg["a_max"]),
            cfg["n_a_detail"])
        A_sweep = np.logspace(
            np.log10(cfg["a_min"]), np.log10(cfg["a_max"]),
            cfg["n_a_sweep"])
        RF_grid = np.logspace(
            np.log10(cfg["rf_min"]), np.log10(cfg["rf_max"]),
            cfg["n_rf"])

        N_detail = cfg["n_mc_detail"]
        N_sweep = cfg["n_mc_sweep"]

        total_steps = 1 + len(RF_grid)
        self.sig_status.emit("Generating ECG signal...")

        np.random.seed(0)
        bio = BioEngine(
            ecg_enabled=cfg.get("ecg_enabled", True),
            ecg_rate=cfg.get("ecg_rate", 400.0),
            ecg_amp=cfg.get("ecg_amp", 1.5e-3),
            emg_enabled=False,
            lead_gain=cfg.get("lead_gain", 1.0))
        v_ecg, _ = bio.generate(duration, internal_fs)
        V_fft = np.fft.rfft(v_ecg)

        freqs = np.fft.rfftfreq(n_hr, d=1.0 / internal_fs)
        omega = 2.0 * np.pi * freqs

        if self._cancelled:
            self.sig_status.emit("Cancelled.")
            return

        # Panel (a): bare metal
        self.sig_status.emit("Panel (a): bare metal sweep...")
        rc_bare, re_bare = _run_area_sweep(
            f=1.0, R_f_sp=cfg["coated_specific"].get("Rf", 362.0),
            A_grid=A_detail, N_real=N_detail,
            V_fft=V_fft, omega=omega, n_hr=n_hr,
            cfg=cfg, r_shunt=r_shunt, seed_offset=0)
        med_bare = np.median(rc_bare, axis=1)

        self.sig_progress.emit(1, total_steps)
        elapsed = time.time() - t0
        self.sig_status.emit(
            f"Panel (a) done ({elapsed:.0f}s). Starting Rf sweep...")

        if self._cancelled:
            self.sig_status.emit("Cancelled.")
            return

        # Panel (b): Rf sweep
        snr_map = np.zeros((len(A_sweep), len(RF_grid)))
        corr_map = np.zeros_like(snr_map)
        ecg_map = np.zeros_like(snr_map)

        for irf, Rf in enumerate(RF_grid):
            if self._cancelled:
                self.sig_status.emit("Cancelled.")
                return

            rc, re = _run_area_sweep(
                f=0.0, R_f_sp=float(Rf),
                A_grid=A_sweep, N_real=N_sweep,
                V_fft=V_fft, omega=omega, n_hr=n_hr,
                cfg=cfg, r_shunt=r_shunt,
                seed_offset=(irf + 1) * 100000)

            med = np.median(rc, axis=1)
            corr_map[:, irf] = med
            ecg_map[:, irf] = re
            floor = np.maximum(re, amplifier_noise)
            snr_map[:, irf] = med / floor

            done = irf + 2
            self.sig_progress.emit(done, total_steps)
            elapsed = time.time() - t0
            pct = done / total_steps * 100
            eta = (total_steps - done) * elapsed / max(done, 1)
            self.sig_status.emit(
                f"Rf sweep [{irf+1}/{len(RF_grid)}]  "
                f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s ({pct:.0f}%)")

        elapsed = time.time() - t0
        self.sig_status.emit(
            f"Done: {elapsed:.1f}s ({elapsed/60:.1f} min). Rendering...")

        result = {
            "rc_bare": rc_bare, "re_bare": re_bare, "med_bare": med_bare,
            "snr_map": snr_map, "corr_map": corr_map, "ecg_map": ecg_map,
            "A_detail": A_detail, "A_sweep": A_sweep, "RF_grid": RF_grid,
        }
        self.sig_finished.emit(result)


# ═══════════════════════════════════════════════════════════════════════════
# OptimizationTab (main widget)
# ═══════════════════════════════════════════════════════════════════════════

class OptimizationTab(QWidget):
    """Electrode area / coating resistance optimisation study tab."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.mw = main_window
        self._worker = None
        self._result = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ── Left panel: controls ──────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(310)
        left_scroll.setMaximumWidth(380)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(6)

        # --- Metal Interface ---
        grp_interface = QGroupBox("Metal Interface")
        iface_layout = QFormLayout()

        self.combo_interface = QComboBox()
        for key, preset in PRESETS.items():
            rct = preset.defaults.get("Rct", 0)
            tag = ""
            if key.startswith("bare"):
                tag = " [bare]"
            elif key.startswith("coated"):
                tag = " [coated]"
            self.combo_interface.addItem(
                f"{preset.name}  (Rct={rct:.0f}){tag}", key)
        idx_i = self.combo_interface.findData("bare_early")
        if idx_i >= 0:
            self.combo_interface.setCurrentIndex(idx_i)
        iface_layout.addRow("Metal preset:", self.combo_interface)

        self.chk_use_main_page = QCheckBox("Use interface from main page")
        self.chk_use_main_page.setChecked(False)
        iface_layout.addRow(self.chk_use_main_page)

        grp_interface.setLayout(iface_layout)
        left_layout.addWidget(grp_interface)

        # --- Coating Film ---
        grp_film = QGroupBox("Protective Film")
        film_layout = QFormLayout()

        self.spin_rf = QDoubleSpinBox()
        self.spin_rf.setRange(1.0, 100000.0)
        self.spin_rf.setValue(362.0)
        self.spin_rf.setSuffix("  \u03a9\u00b7cm\u00b2")
        self.spin_rf.setDecimals(1)
        self.spin_rf.setSingleStep(10.0)
        film_layout.addRow("Rf:", self.spin_rf)

        self.spin_y0f = QDoubleSpinBox()
        self.spin_y0f.setRange(0.1, 50000.0)
        self.spin_y0f.setValue(800.0)
        self.spin_y0f.setSuffix("  \u00b5F/cm\u00b2")
        self.spin_y0f.setDecimals(1)
        film_layout.addRow("Y0_f:", self.spin_y0f)

        self.spin_nf = QDoubleSpinBox()
        self.spin_nf.setRange(0.01, 1.0)
        self.spin_nf.setValue(0.70)
        self.spin_nf.setDecimals(2)
        self.spin_nf.setSingleStep(0.05)
        film_layout.addRow("n_f:", self.spin_nf)

        grp_film.setLayout(film_layout)
        left_layout.addWidget(grp_film)

        # --- Physical Parameters snapshot ---
        grp_snapshot = QGroupBox("Physical Parameters (from main page)")
        snap_layout = QVBoxLayout()
        self.lbl_snapshot = QLabel("(click Run to capture)")
        self.lbl_snapshot.setWordWrap(True)
        self.lbl_snapshot.setStyleSheet(
            "color: #444; font-size: 10px; background: #f8f8f0; "
            "border: 1px solid #ccc; border-radius: 4px; padding: 6px;")
        snap_layout.addWidget(self.lbl_snapshot)
        grp_snapshot.setLayout(snap_layout)
        left_layout.addWidget(grp_snapshot)

        # --- Sweep Parameters ---
        grp_sweep = QGroupBox("Sweep Parameters")
        sweep_layout = QFormLayout()

        self.spin_a_min = QDoubleSpinBox()
        self.spin_a_min.setRange(0.0001, 10.0)
        self.spin_a_min.setValue(0.001)
        self.spin_a_min.setDecimals(4)
        self.spin_a_min.setSuffix(" cm\u00b2")
        sweep_layout.addRow("A min:", self.spin_a_min)

        self.spin_a_max = QDoubleSpinBox()
        self.spin_a_max.setRange(0.01, 100.0)
        self.spin_a_max.setValue(10.0)
        self.spin_a_max.setDecimals(2)
        self.spin_a_max.setSuffix(" cm\u00b2")
        sweep_layout.addRow("A max:", self.spin_a_max)

        self.spin_rf_min = QDoubleSpinBox()
        self.spin_rf_min.setRange(0.1, 1000.0)
        self.spin_rf_min.setValue(1.0)
        self.spin_rf_min.setDecimals(1)
        self.spin_rf_min.setSuffix("  \u03a9\u00b7cm\u00b2")
        sweep_layout.addRow("Rf min:", self.spin_rf_min)

        self.spin_rf_max = QDoubleSpinBox()
        self.spin_rf_max.setRange(10.0, 100000.0)
        self.spin_rf_max.setValue(10000.0)
        self.spin_rf_max.setDecimals(0)
        self.spin_rf_max.setSuffix("  \u03a9\u00b7cm\u00b2")
        sweep_layout.addRow("Rf max:", self.spin_rf_max)

        self.spin_n_rf = QSpinBox()
        self.spin_n_rf.setRange(5, 60)
        self.spin_n_rf.setValue(25)
        self.spin_n_rf.setToolTip("Number of Rf grid points (log-spaced)")
        sweep_layout.addRow("Rf grid points:", self.spin_n_rf)

        self.spin_n_a_detail = QSpinBox()
        self.spin_n_a_detail.setRange(10, 80)
        self.spin_n_a_detail.setValue(35)
        self.spin_n_a_detail.setToolTip("Area grid points for panel (a)")
        sweep_layout.addRow("A points (detail):", self.spin_n_a_detail)

        self.spin_n_a_sweep = QSpinBox()
        self.spin_n_a_sweep.setRange(10, 60)
        self.spin_n_a_sweep.setValue(30)
        self.spin_n_a_sweep.setToolTip("Area grid points for panel (b)")
        sweep_layout.addRow("A points (sweep):", self.spin_n_a_sweep)

        grp_sweep.setLayout(sweep_layout)
        left_layout.addWidget(grp_sweep)

        # --- MC Parameters ---
        grp_mc = QGroupBox("Monte Carlo")
        mc_layout = QFormLayout()

        self.spin_n_mc_detail = QSpinBox()
        self.spin_n_mc_detail.setRange(5, 100)
        self.spin_n_mc_detail.setValue(25)
        self.spin_n_mc_detail.setToolTip("Realisations per A for panel (a)")
        mc_layout.addRow("N_MC (detail):", self.spin_n_mc_detail)

        self.spin_n_mc_sweep = QSpinBox()
        self.spin_n_mc_sweep.setRange(3, 50)
        self.spin_n_mc_sweep.setValue(12)
        self.spin_n_mc_sweep.setToolTip("Realisations per (Rf, A) for panel (b)")
        mc_layout.addRow("N_MC (sweep):", self.spin_n_mc_sweep)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(1.0, 60.0)
        self.spin_duration.setValue(10.0)
        self.spin_duration.setSuffix(" s")
        mc_layout.addRow("Duration:", self.spin_duration)

        grp_mc.setLayout(mc_layout)
        left_layout.addWidget(grp_mc)

        # --- Run / Cancel / Progress ---
        self.btn_run = QPushButton("Run Optimisation")
        self.btn_run.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self._on_run)
        left_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(
            "background-color: #f44336; color: white; padding: 6px;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        left_layout.addWidget(self.btn_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px;")
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch()

        # --- Export ---
        grp_io = QGroupBox("Export")
        io_layout = QHBoxLayout()

        self.btn_export_npz = QPushButton("Export NPZ")
        self.btn_export_npz.setEnabled(False)
        self.btn_export_npz.clicked.connect(self._on_export_npz)
        io_layout.addWidget(self.btn_export_npz)

        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_png.setEnabled(False)
        self.btn_export_png.clicked.connect(self._on_export_png)
        io_layout.addWidget(self.btn_export_png)

        grp_io.setLayout(io_layout)
        left_layout.addWidget(grp_io)

        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll)

        # ── Right panel: results ──────────────────────────────────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.result_tabs = QTabWidget()

        self.canvas_main = FigureCanvas(Figure(figsize=(10, 5)))
        self.result_tabs.addTab(self.canvas_main, "Optimisation")

        right_splitter.addWidget(self.result_tabs)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setFontFamily("Consolas")
        self.txt_summary.setFontPointSize(9)
        self.txt_summary.setMinimumHeight(100)
        self.txt_summary.setPlaceholderText(
            "Run the optimisation study to see results here...")
        right_splitter.addWidget(self.txt_summary)

        right_splitter.setStretchFactor(0, 8)
        right_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(right_splitter, stretch=1)

        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._refresh_snapshot)

    # ── Helpers ────────────────────────────────────────────────────

    def _read_main_window_params(self):
        mw = self.mw
        if mw is None:
            return {}
        r_contact = 10.0
        rho = mw.spin_rho.value()
        area_main = mw.spin_area.value()
        rs_auto = mw.spin_rs.value()
        electrode_dist = max(
            (rs_auto - r_contact) * area_main / max(rho, 0.01), 0.01)

        # Compute lead_gain from electrode positions (same dipole model as compositor)
        lead_gain = 1.0
        try:
            pos = mw.electrode_widget.get_positions()
            w = np.array(pos["working"])
            c = np.array(pos["counter"])
            heart_pos = np.array([0.5, 0.5])
            dipole_axis = np.array([0.0, -1.0])

            midpoint = (w + c) / 2.0
            r_vec = midpoint - heart_pos
            r = max(np.linalg.norm(r_vec), 0.05)
            d_vec = w - c
            d = max(np.linalg.norm(d_vec), 0.01)
            cos_theta = abs(np.dot(d_vec / d, dipole_axis))
            cos_theta = max(cos_theta, 0.05)
            ref_gain = 0.4 / (0.15 ** 2)
            lead_gain = (d * cos_theta / (r ** 2)) / ref_gain
        except Exception:
            pass
        # Override with 3D phantom gain if available (main_3d.py)
        if hasattr(mw, '_cached_lead_gain'):
            lead_gain = mw._cached_lead_gain

        return {
            "r_shunt": mw.spin_rshunt.value(),
            "rho_tissue": rho,
            "electrode_dist": electrode_dist,
            "r_contact": r_contact,
            "internal_fs": mw.spin_internal_fs.value(),
            "adc_lsb": mw.spin_adc_lsb.value() * 1e-6,
            "amplifier_noise": mw.spin_white_noise.value() * 1e-6,
            "ecg_enabled": mw.chk_ecg.isChecked(),
            "ecg_rate": mw.spin_hr.value(),
            "ecg_amp": mw.spin_ecg_amp.value() * 1e-3,
            "lead_gain": lead_gain,
        }

    def _refresh_snapshot(self):
        try:
            p = self._read_main_window_params()
            if not p:
                p = {"r_shunt": 10.0, "rho_tissue": 100.0,
                     "electrode_dist": 0.5, "r_contact": 10.0,
                     "internal_fs": 10000,
                     "adc_lsb": 0.05e-6, "amplifier_noise": 0.05e-6,
                     "ecg_enabled": True, "ecg_rate": 400.0,
                     "ecg_amp": 1.5e-3, "lead_gain": 1.0}
            lines = [
                f"R_shunt = {p['r_shunt']:.1f} \u03a9",
                f"\u03c1 = {p['rho_tissue']:.0f} \u03a9\u00b7cm,  "
                f"dist = {p['electrode_dist']:.2f} cm",
                f"Amp noise = {p['amplifier_noise']*1e6:.3f} \u00b5A",
                f"ECG: {'ON' if p.get('ecg_enabled') else 'OFF'}  "
                f"({p['ecg_rate']:.0f} BPM, "
                f"{p['ecg_amp']*1e3:.1f} mV, "
                f"gain={p.get('lead_gain', 1.0):.3f})",
            ]
            self.lbl_snapshot.setText("\n".join(lines))
        except Exception:
            pass

    def _build_cfg(self):
        mw_params = self._read_main_window_params()
        if not mw_params:
            mw_params = {
                "r_shunt": 10.0, "rho_tissue": 100.0,
                "electrode_dist": 0.5, "r_contact": 10.0,
                "internal_fs": 10000,
                "adc_lsb": 0.05e-6, "amplifier_noise": 0.05e-6,
                "ecg_enabled": True, "ecg_rate": 400.0,
                "ecg_amp": 1.5e-3, "lead_gain": 1.0,
            }

        if self.chk_use_main_page.isChecked() and self.mw is not None:
            cb = self.mw.circuit_builder
            all_sp = cb.get_specific_params()
            topo = cb.get_topology()
            interface_specific = {
                k: v for k, v in all_sp.items()
                if k in ("Rct", "Y0_dl", "n_dl", "sigma_w", "L", "RL")}
            has_warburg = topo.get("has_warburg", False)
            has_inductive = topo.get("has_inductive", False)
            if topo.get("has_film", False):
                film_specific = {
                    "Rf": all_sp.get("Rf", self.spin_rf.value()),
                    "Y0_f": all_sp.get("Y0_f", self.spin_y0f.value()),
                    "n_f": all_sp.get("n_f", self.spin_nf.value()),
                }
            else:
                film_specific = {
                    "Rf": self.spin_rf.value(),
                    "Y0_f": self.spin_y0f.value(),
                    "n_f": self.spin_nf.value(),
                }
        else:
            iface_key = self.combo_interface.currentData()
            iface_preset = PRESETS.get(iface_key, PRESETS["bare_early"])
            active = get_active_param_names(
                iface_preset.has_film, iface_preset.has_warburg,
                iface_preset.has_inductive)
            interface_specific = {k: iface_preset.defaults[k] for k in active}
            has_warburg = iface_preset.has_warburg
            has_inductive = iface_preset.has_inductive
            film_specific = {
                "Rf": self.spin_rf.value(),
                "Y0_f": self.spin_y0f.value(),
                "n_f": self.spin_nf.value(),
            }

        coated_specific = dict(interface_specific)
        coated_specific.update(film_specific)
        coated_topo = {
            "has_film": True,
            "has_warburg": has_warburg,
            "has_inductive": has_inductive,
        }

        self._refresh_snapshot()

        cfg = dict(mw_params)
        cfg.update({
            "coated_specific": coated_specific,
            "coated_topo": coated_topo,
            "duration": self.spin_duration.value(),
            "a_min": self.spin_a_min.value(),
            "a_max": self.spin_a_max.value(),
            "rf_min": self.spin_rf_min.value(),
            "rf_max": self.spin_rf_max.value(),
            "n_rf": self.spin_n_rf.value(),
            "n_a_detail": self.spin_n_a_detail.value(),
            "n_a_sweep": self.spin_n_a_sweep.value(),
            "n_mc_detail": self.spin_n_mc_detail.value(),
            "n_mc_sweep": self.spin_n_mc_sweep.value(),
        })
        return cfg

    # ── Run / Cancel ──────────────────────────────────────────────

    def _on_run(self):
        cfg = self._build_cfg()

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_export_npz.setEnabled(False)
        self.btn_export_png.setEnabled(False)
        self.progress_bar.setValue(0)

        self._worker = OptimizationWorker(cfg, parent=self)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_status.connect(self._on_status)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _on_progress(self, done, total):
        pct = int(done / max(total, 1) * 100)
        self.progress_bar.setValue(pct)

    def _on_status(self, msg):
        self.lbl_status.setText(msg)

    def _on_finished(self, result):
        self._result = result
        cfg = self._worker.cfg

        try:
            fig = _make_optimization_figure(
                result["rc_bare"], result["re_bare"], result["med_bare"],
                result["snr_map"],
                result["A_detail"], result["A_sweep"], result["RF_grid"],
                cfg["amplifier_noise"])

            old = self.result_tabs.widget(0)
            new_canvas = FigureCanvas(fig)
            new_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Expanding)
            self.result_tabs.removeTab(0)
            self.result_tabs.insertTab(0, new_canvas, "Optimisation")
            self.result_tabs.setCurrentIndex(0)
            if old is not None:
                old.deleteLater()
            self.canvas_main = new_canvas
        except Exception as e:
            self.lbl_status.setText(f"Plot error: {e}")

        # Summary text
        try:
            summary = self._build_summary(result, cfg)
            self.txt_summary.setPlainText(summary)
        except Exception as e:
            self.txt_summary.setPlainText(f"Error: {e}")

        self.btn_export_npz.setEnabled(True)
        self.btn_export_png.setEnabled(True)
        self.lbl_status.setText("Optimisation complete.")

    def _on_worker_done(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _build_summary(self, result, cfg):
        lines = []
        lines.append("=" * 70)
        lines.append("  ELECTRODE OPTIMISATION STUDY — SUMMARY")
        lines.append("=" * 70)

        uA = 1e6
        med = result["med_bare"]
        ecg = result["re_bare"]
        A_d = result["A_detail"]
        amp = cfg["amplifier_noise"]

        above = (med > ecg) & (med > amp)
        if np.any(above):
            idx_lo = np.where(above)[0][0]
            idx_hi = np.where(above)[0][-1]
            lines.append(f"\n  Bare metal working window:")
            lines.append(f"    A_min = {A_d[idx_lo]:.4f} cm^2  "
                         f"(corr > floor)")
            lines.append(f"    A_max = {A_d[idx_hi]:.4f} cm^2  "
                         f"(corr > ECG)")
            lines.append(f"    Window spans {np.log10(A_d[idx_hi]/A_d[idx_lo]):.1f} decades")
        else:
            lines.append("\n  No working window for bare metal!")

        lines.append(f"\n  At A = 0.2 cm^2 (typical implant):")
        idx_02 = np.argmin(np.abs(A_d - 0.2))
        lines.append(f"    Corrosion RMS = {med[idx_02]*uA:.3f} uA")
        lines.append(f"    ECG RMS       = {ecg[idx_02]*uA:.3f} uA")
        lines.append(f"    Amp floor     = {amp*uA:.3f} uA")
        lines.append(f"    SNR           = {med[idx_02]/max(ecg[idx_02], amp):.2f}")

        # Rf sweep summary
        snr_map = result["snr_map"]
        RF = result["RF_grid"]
        A_s = result["A_sweep"]

        lines.append(f"\n  Phase diagram: {len(RF)} x {len(A_s)} grid")
        lines.append(f"    Rf range: {RF[0]:.1f} — {RF[-1]:.0f} Ohm*cm^2")
        lines.append(f"    A  range: {A_s[0]:.4f} — {A_s[-1]:.1f} cm^2")

        frac_good = np.mean(snr_map > 1) * 100
        lines.append(f"    SNR > 1 fraction: {frac_good:.1f}% of parameter space")

        # Per Rf: window bounds
        lines.append(f"\n  {'Rf':>10}  {'A_min':>10}  {'A_max':>10}  {'Window':>10}")
        lines.append("  " + "-" * 45)
        for irf in range(0, len(RF), max(1, len(RF) // 8)):
            col = snr_map[:, irf]
            good = np.where(col > 1)[0]
            if len(good) > 0:
                a_lo = A_s[good[0]]
                a_hi = A_s[good[-1]]
                dec = np.log10(a_hi / a_lo)
                lines.append(f"  {RF[irf]:>10.1f}  {a_lo:>10.4f}  "
                             f"{a_hi:>10.4f}  {dec:>8.1f} dec")
            else:
                lines.append(f"  {RF[irf]:>10.1f}  {'—':>10}  "
                             f"{'—':>10}  {'none':>10}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    # ── Export ─────────────────────────────────────────────────────

    def _on_export_npz(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save NPZ", "electrode_optimization_cache.npz",
            "NPZ Files (*.npz)")
        if path:
            np.savez(path, **self._result)
            self.lbl_status.setText(f"Exported to {path}")

    def _on_export_png(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "electrode_optimization.png",
            "PNG Files (*.png);;SVG Files (*.svg)")
        if path:
            if self.canvas_main and self.canvas_main.figure:
                self.canvas_main.figure.savefig(
                    path, dpi=200, bbox_inches="tight")
                self.lbl_status.setText(f"Exported to {path}")
