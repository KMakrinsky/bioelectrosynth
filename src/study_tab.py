"""
Coating Breach Detection Study — interactive GUI tab.

Integrates the Monte Carlo coating-breach study into the main
BioElectroSynth GUI.

Physical model:  A coated electrode has a film layer (Rf, CPEf) on top
of the metal–electrolyte interface (Rct, CPEdl).  Coating breach means
the film is removed from a fraction of the surface, while the underlying
interface stays the same.  The breached (bare) zone has NO film but the
SAME Rct, CPEdl as the intact zone.

Results (regression, area comparison, point-wise plots) render inside
the tab using embedded matplotlib canvases.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

from scipy import stats, signal as sp_signal
from scipy.stats import kurtosis as scipy_kurtosis

import matplotlib
matplotlib.use("Agg")  # ensure non-interactive backend is available
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QScrollArea,
    QTabWidget, QTextEdit, QComboBox, QLineEdit, QProgressBar,
    QFileDialog, QSplitter, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from .compositor import SignalCompositor
from .circuit_model import (
    build_electrode_circuit, convert_to_absolute,
    PRESETS, get_active_param_names, Parallel,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: compute metrics
# ═══════════════════════════════════════════════════════════════════════════

def _chebyshev_basis(N: int) -> np.ndarray:
    """Orthonormal discrete Chebyshev polynomials (Gram) for N points.

    Returns C of shape (N, N) where row k is the k-th basis vector.
    """
    x = np.arange(N, dtype=float)
    tnx = np.zeros((N, N), dtype=float)
    tnx[0, :] = 1.0
    if N > 1:
        tnx[1, :] = 2.0 * x - (N - 1)
    for nn in range(1, N - 1):
        tnx[nn + 1, :] = (
            (2 * nn + 1) * (2.0 * x - N + 1) * tnx[nn, :] -
            nn * (N * N - nn * nn) * tnx[nn - 1, :]
        ) / (nn + 1)
    # Norm: h_n = sum(t_n^2)
    h = np.sqrt(np.sum(tnx ** 2, axis=1))
    h[h == 0] = 1.0
    return tnx / h[:, None]


# Module-level cache for Chebyshev basis
_CHEB_CACHE = {}


def _compute_metrics(sig: np.ndarray, fs: float) -> dict:
    """Extract EN metrics from a ZRA current signal.

    Metrics
    -------
    rms            : root mean square of the full signal
    rms_ma1s       : RMS of 1-second moving average (low-freq drift power)
    rms_resid1s    : RMS of (signal − moving_avg_1s) (high-freq residual)
    spectral_slope : PSD log-log slope (characterises noise type)
    kurtosis       : Fisher kurtosis (sensitive to transients / bubbles)
    band_power     : integrated PSD in 0.1-10 Hz band
    cheb_low       : Chebyshev low-order power (k=2..5) — sensitive to
                     corrosion drift / 1/f^2 noise, rejects instrument noise
    cheb_ratio     : cheb_low / total Chebyshev power — normalised version
    """
    n = len(sig)
    rms = float(np.sqrt(np.mean(sig ** 2)))

    # ── Moving-average decomposition (1-second window) ────────────────
    win = max(int(fs * 1.0), 1)  # 1 s window in samples
    if win >= n:
        ma = np.full(n, np.mean(sig))
    else:
        # Causal uniform filter (cumsum trick), then centre-align
        cs = np.cumsum(np.insert(sig, 0, 0.0))
        ma_raw = (cs[win:] - cs[:-win]) / win
        # Pad edges to keep length = n
        pad_left = win // 2
        pad_right = n - len(ma_raw) - pad_left
        ma = np.concatenate([
            np.full(pad_left, ma_raw[0]),
            ma_raw,
            np.full(max(pad_right, 0), ma_raw[-1]),
        ])[:n]
    resid = sig - ma
    rms_ma1s = float(np.sqrt(np.mean(ma ** 2)))
    rms_resid1s = float(np.sqrt(np.mean(resid ** 2)))

    nyq = fs / 2.0
    nperseg = min(n, max(int(n * 0.5), 4))
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)

    mask = freqs > 0
    if np.sum(mask) > 3:
        log_f = np.log10(freqs[mask])
        log_p = np.log10(np.maximum(psd[mask], 1e-40))
        slope, *_ = np.polyfit(log_f, log_p, 1)
    else:
        slope = np.nan

    kurt = float(scipy_kurtosis(sig, fisher=True))

    f_lo = max(0.1, freqs[1] if len(freqs) > 1 else 0.1)
    f_hi = min(10.0, nyq * 0.95)
    if f_hi > f_lo:
        band = (freqs >= f_lo) & (freqs <= f_hi)
        band_pwr = float(np.trapz(psd[band], freqs[band])) if np.sum(band) > 1 else 0.0
    else:
        band_pwr = 0.0

    # ── Chebyshev noise spectroscopy ──────────────────────────────────
    # Split signal into non-overlapping segments of length L, project
    # each onto discrete Chebyshev basis, average intensities Y_k^2.
    # Low-order lines (k=2..5) capture corrosion-rate fluctuations
    # while instrument white noise is spread uniformly across all k.
    L = min(n, max(int(fs * 2), 8))  # segment length ~2 s or full signal
    if L not in _CHEB_CACHE:
        _CHEB_CACHE[L] = _chebyshev_basis(L)
    C = _CHEB_CACHE[L]

    n_segs = max(n // L, 1)
    cheb_low = 0.0
    cheb_total = 0.0
    k_lo, k_hi = 2, min(6, L)  # indices 2..5

    for s in range(n_segs):
        seg = sig[s * L: (s + 1) * L]
        if len(seg) < L:
            break
        Y = C @ seg  # Chebyshev coefficients
        intensities = Y ** 2
        cheb_low += float(np.sum(intensities[k_lo:k_hi]))
        cheb_total += float(np.sum(intensities[k_lo:]))  # skip k=0,1 (DC, linear trend)

    cheb_low /= max(n_segs, 1)
    cheb_total /= max(n_segs, 1)
    cheb_ratio = cheb_low / max(cheb_total, 1e-40)

    return dict(rms=rms, rms_ma1s=rms_ma1s, rms_resid1s=rms_resid1s,
                spectral_slope=slope, kurtosis=kurt,
                band_power=band_pwr, cheb_low=cheb_low, cheb_ratio=cheb_ratio)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: OLS regression
# ═══════════════════════════════════════════════════════════════════════════

def _ols_fit(x, y):
    """OLS linear fit  y = b0 + b1*x  with full stats."""
    res = stats.linregress(x, y)
    n = len(x)
    y_hat = res.intercept + res.slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    s_resid = np.sqrt(ss_res / max(n - 2, 1))
    return dict(
        b0=res.intercept, b1=res.slope,
        p_slope=res.pvalue, r2=res.rvalue ** 2,
        se_slope=res.stderr,
        s_resid=s_resid, n=n,
        x_mean=float(np.mean(x)),
        SS_x=float(np.sum((x - np.mean(x)) ** 2)),
    )


def _predict(fit, f_arr, alpha=0.05, kind="prediction"):
    """Compute fitted values + interval band."""
    y_hat = fit["b0"] + fit["b1"] * f_arr
    n = fit["n"]
    t_val = stats.t.ppf(1 - alpha / 2, n - 2)
    h = 1.0 / n + (f_arr - fit["x_mean"]) ** 2 / fit["SS_x"]
    if kind == "prediction":
        se = fit["s_resid"] * np.sqrt(1.0 + h)
    else:
        se = fit["s_resid"] * np.sqrt(h)
    return y_hat, y_hat - t_val * se, y_hat + t_val * se


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build mixed circuit  (coated || bare)
# ═══════════════════════════════════════════════════════════════════════════

def _build_mixed_circuit(f: float, area: float,
                         coated_specific: dict, coated_topo: dict,
                         rho_tissue: float, electrode_dist: float,
                         r_contact: float):
    """
    Build electrode equivalent circuit for a partially degraded coating.

    Physical model
    --------------
    The electrode surface has two zones sharing the **same** metal–electrolyte
    interface (Rct, CPEdl) but differing in whether the protective film/coating
    (Rf, CPEf) is intact.

        Coated zone  (area × (1−f)):  Film — Interface
        Bare zone    (area × f):      Interface only  (film removed by breach)

    Both zones are connected in parallel (same electrolyte path).

    Parameters
    ----------
    f              : fraction of surface where the coating is breached (0..1)
    area           : total electrode area, cm²
    coated_specific: specific (per-cm²) parameter values — MUST include film
                     params (Rf, Y0_f, n_f) plus interface (Rct, Y0_dl, n_dl)
    coated_topo    : dict with has_film, has_warburg, has_inductive
    """
    R_s = r_contact + (rho_tissue * electrode_dist / max(area, 1e-4))

    # Interface-only specific params (same metal in both zones)
    interface_keys = {"Rct", "Y0_dl", "n_dl"}
    # Also keep Warburg/inductive params if present
    if coated_topo.get("has_warburg"):
        interface_keys.add("sigma_w")
    if coated_topo.get("has_inductive"):
        interface_keys |= {"L", "RL"}

    interface_specific = {k: v for k, v in coated_specific.items()
                          if k in interface_keys}

    # --- Pure coated (f <= 0) — full circuit with film ---
    if f <= 0.0:
        ab = convert_to_absolute(coated_specific, area)
        circuit, R_ct, C_dl = build_electrode_circuit(
            ab, coated_topo["has_film"],
            coated_topo.get("has_warburg", False),
            coated_topo.get("has_inductive", False),
        )
        return circuit, R_ct, C_dl, R_s

    # --- Pure bare (f >= 1) — interface only, no film ---
    if f >= 1.0:
        ab = convert_to_absolute(interface_specific, area)
        circuit, R_ct, C_dl = build_electrode_circuit(
            ab, has_film=False,
            has_warburg=coated_topo.get("has_warburg", False),
            has_inductive=coated_topo.get("has_inductive", False),
        )
        return circuit, R_ct, C_dl, R_s

    # --- Mixed surface ---
    area_coated = (1.0 - f) * area
    area_bare = f * area

    # Coated zone: full circuit WITH film
    ab_c = convert_to_absolute(coated_specific, area_coated)
    circuit_coated, Rct_coated, Cdl_coated = build_electrode_circuit(
        ab_c, coated_topo["has_film"],
        coated_topo.get("has_warburg", False),
        coated_topo.get("has_inductive", False),
    )

    # Bare zone: SAME interface, NO film
    ab_b = convert_to_absolute(interface_specific, area_bare)
    circuit_bare, Rct_bare, Cdl_bare = build_electrode_circuit(
        ab_b, has_film=False,
        has_warburg=coated_topo.get("has_warburg", False),
        has_inductive=coated_topo.get("has_inductive", False),
    )

    circuit = Parallel(circuit_coated, circuit_bare)
    R_ct_eff = 1.0 / (1.0 / Rct_coated + 1.0 / Rct_bare)
    C_dl_eff = Cdl_coated + Cdl_bare

    return circuit, R_ct_eff, C_dl_eff, R_s


# ═══════════════════════════════════════════════════════════════════════════
# Top-level worker function (must be picklable for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

# We store the study config in a module-level variable set before the pool
# launches, so that each worker can access it without passing unpicklable
# objects through the argument tuple.
_STUDY_CFG = {}


def _run_single_sim(args):
    """One Monte Carlo realisation.  args = (f, area, target_fs, seed)."""
    f, area, target_fs, seed = args
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    cfg = _STUDY_CFG

    circuit, R_ct_eff, C_dl_eff, R_s = _build_mixed_circuit(
        f, area,
        cfg["coated_specific"], cfg["coated_topo"],
        cfg["rho_tissue"], cfg["electrode_dist"], cfg["r_contact"],
    )

    # ── Effective polarisation resistance for Stern-Geary ──────────────
    #
    # Stern-Geary:  I_corr = B / Rp   where Rp = total polarisation
    # resistance at DC.
    #
    # For mixed surface:
    #   coated zone: Rp_coated = Rf + Rct   (film + interface, DC)
    #   bare zone:   Rp_bare   = Rct         (interface only)
    #
    #   1/Rp_eff = (1-f)*A / Rp_coated_abs  +  f*A / Rp_bare_abs
    #   where Rp_abs = Rp_specific / area_zone
    #
    # Simplification (same Rct_specific everywhere):
    #   Rp_eff_specific = 1 / [(1-f)/Rp_coated + f/Rp_bare]
    #
    # We pass Rp_eff as the R_ct for Stern-Geary computation, because
    # physics_aware_mode derives I_corr = B / R_ct.  For a coated electrode,
    # the total polarisation resistance (not just charge-transfer) determines
    # the corrosion rate.
    sp = cfg["coated_specific"]
    Rct_sp = sp.get("Rct", 300.0)
    Rf_sp = sp.get("Rf", 0.0)
    Rp_coated_sp = Rf_sp + Rct_sp       # specific, Ohm*cm^2
    Rp_bare_sp = Rct_sp                  # specific, Ohm*cm^2

    # Effective specific Rp for the mixed surface
    inv_Rp_eff = (1.0 - f) / max(Rp_coated_sp, 0.01) + f / max(Rp_bare_sp, 0.01)
    Rp_eff_sp = 1.0 / max(inv_Rp_eff, 1e-12)

    # Convert to absolute for the given total area
    Rp_eff_abs = Rp_eff_sp / max(area, 1e-4)

    compositor = SignalCompositor()

    params = {
        "duration":    cfg["duration"],
        "fs":          target_fs,
        "internal_fs": cfg["internal_fs"],
        "adc_lsb":     cfg["adc_lsb"],
        "electrodes": cfg.get("electrode_pos", {
            "working": [0.48, 0.65],
            "counter": [0.52, 0.65],
        }),
        "corrosion": {
            "dc_current":       50e-6,       # overridden by physics_aware
            "noise_power":      15e-6,       # overridden by physics_aware
            "noise_alpha":      2.0,
            "bubble_rate":      5.0,
            "bubble_amp_mean":  0.5e-6,
            "bubble_decay":     0.08,
            "trend_type":       "polynomial",
            "physics_aware_mode":       True,
            "use_circuit_model":        True,
            "use_symmetric_electrodes": True,
            "R_s":              R_s,
            # Pass Rp_eff (not R_ct!) for Stern-Geary computation.
            # This ensures I_corr = B/Rp correctly captures the total
            # protection including the film.
            "R_ct":             Rp_eff_abs,
            "C_dl":             C_dl_eff,
            "R_shunt":          cfg["r_shunt"],
            "electrode_circuit": circuit,
            "R_ct_value":       Rp_eff_abs,
            "C_dl_effective":   C_dl_eff,
        },
        "bio": {
            "ecg_enabled": cfg["ecg_enabled"],
            "ecg_rate":    cfg["ecg_rate"],
            "ecg_amp":     cfg["ecg_amp"],
            "emg_enabled": False,
        },
        "sensor": {
            "white_noise_level": cfg["amplifier_noise"],
            "mains_hum_level":   0.0,
        },
    }

    try:
        result = compositor.generate(params)
        sig = result["total_signal"]
        metrics = _compute_metrics(sig, target_fs)
    except Exception as exc:
        metrics = {k: np.nan for k in METRIC_KEYS}

    metrics.update(f=f, area=area, target_fs=target_fs,
                   seed=seed, R_ct_eff=R_ct_eff)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Regression analysis
# ═══════════════════════════════════════════════════════════════════════════

METRIC_KEYS = ["rms", "rms_ma1s", "rms_resid1s",
               "spectral_slope", "kurtosis", "band_power",
               "cheb_low", "cheb_ratio"]

N_MEASUREMENTS = [1, 2, 5, 10, 20, 30, 50, 100]


def _regression_analysis(raw_df, areas, sampling_rates, p_threshold):
    """OLS regression  metric = b0 + b1*f  for each (area, fs).

    Runs for both RMS and cheb_low metrics.  Chebyshev columns are
    prefixed with ``cheb_`` in the output DataFrame.
    """
    records = []
    z_alpha = stats.norm.ppf(1 - p_threshold / 2)

    for target_fs in sampling_rates:
        for area in areas:
            sub = raw_df[(raw_df["area"] == area) &
                         (raw_df["target_fs"] == target_fs)].copy()
            x = sub["f"].values

            if len(x) < 10:
                continue

            # ── RMS regression ──
            y_rms = sub["rms"].values
            fit = _ols_fit(x, y_rms)
            sigma = fit["s_resid"]
            b1 = fit["b1"]

            f_det_dict = {}
            n_req_10pct = np.nan
            for N in N_MEASUREMENTS:
                if b1 > 0:
                    fd = z_alpha * sigma * np.sqrt(2.0 / N) / b1
                    f_det_dict[N] = fd
                else:
                    f_det_dict[N] = np.nan

            if b1 > 0:
                n_req_10 = (z_alpha * sigma * np.sqrt(2) / (b1 * 0.10)) ** 2
                n_req_10pct = np.ceil(n_req_10)

            rec = dict(
                area=area, target_fs=target_fs,
                b0=fit["b0"], b1=b1,
                b1_uA_per_pct=b1 * 1e6 / 100,
                p_slope=fit["p_slope"],
                se_slope=fit["se_slope"],
                r2=fit["r2"],
                s_resid=sigma,
                n_points=fit["n"],
                n_req_10pct=n_req_10pct,
            )
            for N in N_MEASUREMENTS:
                rec[f"f_det_N{N}"] = f_det_dict.get(N, np.nan)
                rec[f"f_det_N{N}_pct"] = f_det_dict.get(N, np.nan) * 100

            # ── Alternative metric regressions ──
            # For each extra metric: fit OLS, store r2, p_slope, f_det_N30
            _extra_metrics = {
                "cheb":    "cheb_low",      # Chebyshev low-order power
                "ma1s":    "rms_ma1s",      # RMS of 1-s moving average
                "resid1s": "rms_resid1s",   # RMS of (signal − MA)
            }
            for prefix, col in _extra_metrics.items():
                if col in sub.columns and sub[col].notna().sum() > 10:
                    y_alt = sub[col].values
                    fit_alt = _ols_fit(x, y_alt)
                    rec[f"{prefix}_b0"] = fit_alt["b0"]
                    rec[f"{prefix}_b1"] = fit_alt["b1"]
                    rec[f"{prefix}_p_slope"] = fit_alt["p_slope"]
                    rec[f"{prefix}_r2"] = fit_alt["r2"]
                    rec[f"{prefix}_s_resid"] = fit_alt["s_resid"]
                    if fit_alt["b1"] > 0:
                        fd_alt = (z_alpha * fit_alt["s_resid"] *
                                  np.sqrt(2.0 / 30) / fit_alt["b1"])
                        rec[f"{prefix}_f_det_N30"] = fd_alt
                        rec[f"{prefix}_f_det_N30_pct"] = fd_alt * 100
                    else:
                        rec[f"{prefix}_f_det_N30"] = np.nan
                        rec[f"{prefix}_f_det_N30_pct"] = np.nan
                else:
                    for sfx in ["_b1", "_p_slope", "_r2", "_s_resid",
                                "_f_det_N30", "_f_det_N30_pct"]:
                        rec[f"{prefix}{sfx}"] = np.nan

            records.append(rec)

    return pd.DataFrame(records)


def _compute_statistics(df, areas, fractions, sampling_rates, p_threshold):
    """Point-wise statistics (means, CIs, t-tests)."""
    records = []
    for target_fs in sampling_rates:
        for area in areas:
            baseline = df[(df["area"] == area) & (df["f"] == 0.0) &
                          (df["target_fs"] == target_fs)]
            for f in fractions:
                group = df[(df["area"] == area) & (df["f"] == f) &
                           (df["target_fs"] == target_fs)]
                rec = {"area": area, "f": f, "target_fs": target_fs}
                for m in METRIC_KEYS:
                    vals = group[m].dropna().values
                    base_vals = baseline[m].dropna().values
                    if len(vals) < 3 or len(base_vals) < 3:
                        rec[f"{m}_mean"] = np.nan
                        rec[f"{m}_ci_lo"] = np.nan
                        rec[f"{m}_ci_hi"] = np.nan
                        rec[f"{m}_pvalue"] = 1.0
                        continue
                    mean = float(np.mean(vals))
                    sem = float(stats.sem(vals))
                    ci = stats.t.interval(0.95, len(vals) - 1,
                                          loc=mean, scale=max(sem, 1e-30))
                    rec[f"{m}_mean"] = mean
                    rec[f"{m}_ci_lo"] = ci[0]
                    rec[f"{m}_ci_hi"] = ci[1]
                    if f == 0.0:
                        rec[f"{m}_pvalue"] = 1.0
                    else:
                        _, p = stats.ttest_ind(base_vals, vals, equal_var=False)
                        rec[f"{m}_pvalue"] = float(p)
                records.append(rec)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting functions (return Figure objects instead of saving to disk)
# ═══════════════════════════════════════════════════════════════════════════

def _make_regression_figure(raw_df, reg_df, areas, sampling_rates,
                            duration, amplifier_noise, adc_lsb, p_threshold):
    """Six-panel regression figure using the best metric (Resid-1s).

    Falls back to RMS when rms_resid1s is not available.
    """
    # Pick best available metric column
    if "rms_resid1s" in raw_df.columns and raw_df["rms_resid1s"].notna().any():
        y_col, y_label = "rms_resid1s", "RMS residual after MA(1s)"
        reg_pfx = "resid1s_"
    else:
        y_col, y_label = "rms", "RMS current"
        reg_pfx = ""

    fig = Figure(figsize=(18, 13))
    fig.suptitle(
        f"Coating Breach Detection: OLS Regression  "
        f"{y_label} = b0 + b1*f\n"
        f"({duration:.0f}s records, "
        f"amp noise {amplifier_noise*1e6:.2f} uA, "
        f"ADC LSB {adc_lsb*1e6:.2f} uA, "
        f"p < {p_threshold})",
        fontsize=12, fontweight="bold",
    )

    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    colors_area = plt.cm.viridis(np.linspace(0.2, 0.9, len(areas)))
    fs_colors = {1: "tab:blue", 10: "tab:orange", 100: "tab:green"}
    fs_markers = {1: "^", 10: "s", 100: "o"}

    ref_area = areas[len(areas) // 2] if areas else 0.2
    z_alpha = stats.norm.ppf(1 - p_threshold / 2)

    if reg_df is None or len(reg_df) == 0:
        fig.text(0.5, 0.5, "No regression data\n(need more MC realisations)",
                 ha="center", va="center", fontsize=16)
        return fig

    # Use resid1s regression columns if available, else fallback to RMS
    b1_col = f"{reg_pfx}b1" if f"{reg_pfx}b1" in reg_df.columns else "b1"
    sr_col = f"{reg_pfx}s_resid" if f"{reg_pfx}s_resid" in reg_df.columns else "s_resid"
    r2_col = f"{reg_pfx}r2" if f"{reg_pfx}r2" in reg_df.columns else "r2"
    ps_col = f"{reg_pfx}p_slope" if f"{reg_pfx}p_slope" in reg_df.columns else "p_slope"

    reg_df = reg_df.copy()
    reg_df["_b1"] = reg_df[b1_col].fillna(reg_df["b1"])
    reg_df["_sr"] = reg_df[sr_col].fillna(reg_df["s_resid"])
    reg_df["ratio"] = reg_df["_sr"] / reg_df["_b1"].replace(0, np.nan)
    ratio_mean = reg_df["ratio"].mean()
    ratio_std = reg_df["ratio"].std()
    if np.isnan(ratio_mean):
        ratio_mean = 0.22
        ratio_std = 0.02

    # Row 1: scatter + fit + CI (one panel per sampling rate, A = ref_area)
    for col, target_fs in enumerate(sampling_rates[:3]):
        ax = fig.add_subplot(gs[0, col])
        sub = raw_df[(raw_df["area"] == ref_area) &
                     (raw_df["target_fs"] == target_fs)]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        x = sub["f"].values
        y = sub[y_col].values * 1e6

        fit = _ols_fit(x, y)
        f_plot = np.linspace(0, 0.55, 300)
        y_hat, pi_lo, pi_hi = _predict(fit, f_plot, kind="prediction")
        _, ci_lo, ci_hi = _predict(fit, f_plot, kind="confidence")

        ax.scatter(x * 100, y, s=12, alpha=0.30, color="grey", zorder=2,
                   label="MC samples")
        ax.plot(f_plot * 100, y_hat, "-", color="crimson", linewidth=2.5,
                zorder=3,
                label=f"fit: {fit['b0']:.2f} + {fit['b1']:.2f}f")
        ax.fill_between(f_plot * 100, ci_lo, ci_hi, alpha=0.30,
                        color="crimson", label="95% CI (mean)")
        ax.fill_between(f_plot * 100, pi_lo, pi_hi, alpha=0.08,
                        color="royalblue", label="95% PI (single obs)")

        fd30 = z_alpha * ratio_mean * np.sqrt(2.0 / 30) * 100
        ax.axvline(fd30, color="red", ls=":", lw=2,
                   label=f"f_det(N=30) = {fd30:.1f}%")

        ax.set_xlabel("Exposed bare metal (%)")
        ax.set_ylabel(f"{y_label} (uA)")
        ax.set_title(f"A={ref_area} cm^2,  fs={target_fs} Hz\n"
                     f"R^2={fit['r2']:.3f},  p(slope)={fit['p_slope']:.1e}")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3)

    # Row 2 left: Universal f_detect(N) curve
    ax = fig.add_subplot(gs[1, 0])
    N_fine = np.logspace(0, 2.5, 200)
    f_det_mean = z_alpha * ratio_mean * np.sqrt(2.0 / N_fine) * 100
    ax.plot(N_fine, f_det_mean, "-", color="black", linewidth=3,
            label=f"pooled: sigma/b1 = {ratio_mean:.3f}", zorder=5)
    f_det_hi = z_alpha * (ratio_mean + ratio_std) * np.sqrt(2.0 / N_fine) * 100
    f_det_lo = z_alpha * max(ratio_mean - ratio_std, 0.001) * np.sqrt(2.0 / N_fine) * 100
    ax.fill_between(N_fine, f_det_lo, f_det_hi, alpha=0.15, color="grey",
                    label=f"+/- 1 std ({ratio_std:.3f})")

    for _, r in reg_df.iterrows():
        ratio_i = r["ratio"]
        if np.isnan(ratio_i):
            continue
        f_det_i = z_alpha * ratio_i * np.sqrt(2.0 / N_fine) * 100
        ax.plot(N_fine, f_det_i, "-", alpha=0.12, color="steelblue", linewidth=0.8)

    ax.axhline(10, color="red", ls="--", lw=1.5, alpha=0.8, label="10% target")
    ax.axhline(5, color="orange", ls="--", lw=1, alpha=0.5, label="5% target")

    for N_mark in [1, 5, 10, 30, 100]:
        fd = z_alpha * ratio_mean * np.sqrt(2.0 / N_mark) * 100
        if fd <= 100:
            ax.plot(N_mark, fd, "ko", markersize=5, zorder=6)
            ax.annotate(f"{fd:.0f}%", (N_mark, fd),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=7, fontweight="bold")

    ax.set_xlabel(f"N recordings  (each {int(duration)}s)")
    ax.set_ylabel("Min detectable breach (%)")
    ax.set_title(f"Universal detection curve\n"
                 f"f_det(N) = z_a * (sigma/b1) * sqrt(2/N), p<{p_threshold}")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.9, 120)
    ax.set_xscale("log")

    # Row 2 centre: sigma/b1 ratio
    ax = fig.add_subplot(gs[1, 1])
    for target_fs in sampling_rates:
        sub = reg_df[reg_df["target_fs"] == target_fs].sort_values("area")
        c = fs_colors.get(target_fs, "tab:blue")
        m = fs_markers.get(target_fs, "o")
        ax.plot(sub["area"], sub["ratio"],
                f"-{m}", color=c,
                label=f"fs={target_fs} Hz", markersize=8, linewidth=2)
    ax.axhline(ratio_mean, color="black", ls="-", lw=2,
               label=f"mean = {ratio_mean:.3f}")
    ax.axhspan(ratio_mean - ratio_std, ratio_mean + ratio_std,
               alpha=0.15, color="grey",
               label=f"+/- 1 std = {ratio_std:.3f}")
    ax.set_xlabel("Electrode area (cm^2)")
    ax.set_ylabel("sigma / b1  ratio")
    cv = ratio_std / ratio_mean * 100 if ratio_mean > 0 else 0
    ax.set_title(f"Ratio constancy check\n"
                 f"(mean={ratio_mean:.3f}, CV={cv:.1f}%)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Row 2 right: Universal detection table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    N_list = [1, 2, 5, 10, 20, 30, 50, 100]
    lines = [
        f"UNIVERSAL DETECTION TABLE  ({y_label})",
        f"sigma / b1 = {ratio_mean:.3f} +/- {ratio_std:.3f}",
        f"(pooled over {len(reg_df)} configurations)",
        "",
        f"{'N rec':>7}  {'Time':>7}  {'f_det':>8}",
        "-" * 30,
    ]
    for N in N_list:
        fd = z_alpha * ratio_mean * np.sqrt(2.0 / N) * 100
        t_min = N * duration / 60
        lines.append(f"{N:>7d}  {t_min:>5.1f} min  {fd:>7.1f} %")
    lines.append("-" * 30)
    n10 = np.ceil((z_alpha * ratio_mean * np.sqrt(2) / 0.10) ** 2)
    lines.append(f"N for 10% detection: {n10:.0f}")
    lines.append(f"  = {n10 * duration / 60:.1f} min total recording")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.9))
    ax.set_title("Detection thresholds (pooled)")

    fig.tight_layout()
    return fig


def _make_area_comparison_figure(raw_df, reg_df, areas, target_fs,
                                 duration, amplifier_noise, adc_lsb,
                                 n_mc, p_threshold):
    """Area comparison figure at a fixed fs.  Returns Figure."""
    # Pick best metric
    if "rms_resid1s" in raw_df.columns and raw_df["rms_resid1s"].notna().any():
        y_col, y_label = "rms_resid1s", "RMS residual after MA(1s)"
    else:
        y_col, y_label = "rms", "RMS current"

    n_areas = len(areas)
    n_rows = (n_areas + 2) // 3
    n_cols = 3
    fig = Figure(figsize=(18, 5 * n_rows))
    fig.suptitle(
        f"Area Comparison at fs = {target_fs} Hz  |  "
        f"OLS: {y_label} = b0 + b1*f\n"
        f"({duration:.0f}s records, "
        f"amp noise {amplifier_noise*1e6:.2f} uA, "
        f"ADC LSB {adc_lsb*1e6:.2f} uA, "
        f"N_mc = {n_mc}, p < {p_threshold})",
        fontsize=12, fontweight="bold",
    )

    colors_area = plt.cm.viridis(np.linspace(0.15, 0.90, n_areas))
    z_alpha = stats.norm.ppf(1 - p_threshold / 2)

    fits = {}
    axes = []
    for idx in range(n_areas + 1):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        axes.append(ax)

    # Individual area panels
    for idx, area in enumerate(areas):
        ax = axes[idx]
        sub = raw_df[(raw_df["area"] == area) &
                     (raw_df["target_fs"] == target_fs)]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        x = sub["f"].values
        y = sub[y_col].values * 1e6

        fit = _ols_fit(x, y)
        fits[area] = fit

        f_plot = np.linspace(0, 0.55, 300)
        y_hat, pi_lo, pi_hi = _predict(fit, f_plot, kind="prediction")
        _, ci_lo, ci_hi = _predict(fit, f_plot, kind="confidence")

        ax.scatter(x * 100, y, s=14, alpha=0.30, color=colors_area[idx],
                   zorder=2, edgecolors="none")
        ax.plot(f_plot * 100, y_hat, "-", color=colors_area[idx],
                linewidth=2.5, zorder=4)
        ax.fill_between(f_plot * 100, ci_lo, ci_hi, alpha=0.30,
                        color=colors_area[idx])
        ax.fill_between(f_plot * 100, pi_lo, pi_hi, alpha=0.08,
                        color=colors_area[idx])

        ratio = fit["s_resid"] / fit["b1"] if fit["b1"] > 0 else np.nan
        fd30 = z_alpha * ratio * np.sqrt(2.0 / 30) * 100 if not np.isnan(ratio) else np.nan
        n10 = np.ceil((z_alpha * ratio * np.sqrt(2) / 0.10) ** 2) if not np.isnan(ratio) else np.nan

        if not np.isnan(fd30) and fd30 <= 55:
            ax.axvline(fd30, color="red", ls=":", lw=1.5, alpha=0.8)

        info = (f"A = {area} cm^2\n"
                f"b1 = {fit['b1']:.3f} uA/frac\n"
                f"sigma = {fit['s_resid']:.3f} uA\n"
                f"sigma/b1 = {ratio:.3f}\n"
                f"R^2 = {fit['r2']:.3f}\n"
                f"p = {fit['p_slope']:.1e}\n"
                f"f_det(N=30) = {fd30:.1f}%\n"
                f"N for 10% = {n10:.0f}")
        ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=7,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="grey"))
        ax.set_xlabel("Exposed bare metal (%)")
        ax.set_ylabel(f"{y_label} (uA)")
        ax.set_title(f"A = {area} cm^2", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Overlay panel
    ax = axes[n_areas]
    f_plot = np.linspace(0, 0.55, 300)
    for idx, area in enumerate(areas):
        if area not in fits:
            continue
        fit = fits[area]
        y_hat = fit["b0"] + fit["b1"] * f_plot
        _, ci_lo, ci_hi = _predict(fit, f_plot, kind="confidence")
        ax.plot(f_plot * 100, y_hat, "-", color=colors_area[idx],
                linewidth=2.5, label=f"A={area} cm^2", zorder=3)
        ax.fill_between(f_plot * 100, ci_lo, ci_hi, alpha=0.15,
                        color=colors_area[idx])

    noise_floor_uA = np.sqrt(amplifier_noise ** 2 +
                             (adc_lsb / np.sqrt(12)) ** 2) * 1e6
    ax.axhline(noise_floor_uA, color="grey", ls="--", lw=1.5,
               label=f"Noise floor = {noise_floor_uA:.3f} uA")
    ax.set_xlabel("Exposed bare metal (%)")
    ax.set_ylabel(f"{y_label} (uA)")
    ax.set_title(f"All areas: fit lines + 95% CI  (fs = {target_fs} Hz)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Hide unused subplots
    total_panels = n_rows * n_cols
    for idx in range(n_areas + 1, total_panels):
        extra_ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        extra_ax.set_visible(False)

    fig.tight_layout()
    return fig


def _make_pointwise_figure(stats_df, areas, fractions, sampling_rates,
                           duration, amplifier_noise, adc_lsb, p_threshold,
                           raw_df=None, reg_df=None):
    """Six-panel point-wise figure based on REGRESSION fit.

    Heatmap and detection threshold now derive from the OLS regression
    (p_slope, f_det) rather than point-wise t-tests.
    """
    fig = Figure(figsize=(16, 14))
    fig.suptitle(
        "Coating Breach Detection:  Effect of Area, Sampling Rate "
        "& Instrument Noise\n"
        f"(symmetric ZRA, {duration:.0f}s records, "
        f"amplifier {amplifier_noise*1e6:.2f} uA, "
        f"ADC LSB {adc_lsb*1e6:.2f} uA, "
        f"p < {p_threshold})",
        fontsize=11, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30,
                          top=0.92, bottom=0.06, left=0.07, right=0.95)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(areas)))
    fs_styles = {1: "--", 10: "-", 100: "-"}
    fs_markers = {1: "^", 10: "s", 100: "o"}
    z_alpha = stats.norm.ppf(1 - p_threshold / 2)

    # Pick best metric
    if raw_df is not None and "rms_resid1s" in raw_df.columns and raw_df["rms_resid1s"].notna().any():
        y_col, y_label = "rms_resid1s", "RMS resid(1s)"
        reg_pfx = "resid1s_"
    else:
        y_col, y_label = "rms", "RMS current"
        reg_pfx = ""

    # ── Row 1: fit lines per area (first two sampling rates) ──
    for col, target_fs in enumerate(sampling_rates[:2]):
        ax = fig.add_subplot(gs[0, col])
        if raw_df is not None:
            for idx, area in enumerate(areas):
                sub = raw_df[(raw_df["area"] == area) &
                             (raw_df["target_fs"] == target_fs)]
                if len(sub) < 5:
                    continue
                x = sub["f"].values
                y = sub[y_col].values * 1e6
                fit = _ols_fit(x, y)
                f_plot = np.linspace(0, max(x.max(), 0.5), 200)
                y_hat, ci_lo, ci_hi = _predict(fit, f_plot, kind="confidence")
                ax.scatter(x * 100, y, s=8, alpha=0.15, color=colors[idx])
                ax.plot(f_plot * 100, y_hat, "-", color=colors[idx],
                        linewidth=2, label=f"A={area}")
                ax.fill_between(f_plot * 100, ci_lo, ci_hi,
                                alpha=0.12, color=colors[idx])
        else:
            sub_fs = stats_df[stats_df["target_fs"] == target_fs]
            for idx, area in enumerate(areas):
                sub = sub_fs[sub_fs["area"] == area].sort_values("f")
                ax.plot(sub["f"].values * 100, sub["rms_mean"].values * 1e6,
                        "o-", color=colors[idx], label=f"A={area}",
                        markersize=4, linewidth=1.5)
        ax.set_xlabel("Exposed bare metal (%)")
        ax.set_ylabel(f"{y_label} (uA)")
        ax.set_title(f"{y_label} vs degradation, fs = {target_fs} Hz", fontsize=10)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3)

    # ── Row 2 left: Detection threshold from regression f_det ──
    b1_key = f"{reg_pfx}b1" if reg_df is not None and f"{reg_pfx}b1" in reg_df.columns else "b1"
    sr_key = f"{reg_pfx}s_resid" if reg_df is not None and f"{reg_pfx}s_resid" in reg_df.columns else "s_resid"
    ps_key = f"{reg_pfx}p_slope" if reg_df is not None and f"{reg_pfx}p_slope" in reg_df.columns else "p_slope"
    r2_key = f"{reg_pfx}r2" if reg_df is not None and f"{reg_pfx}r2" in reg_df.columns else "r2"

    ax = fig.add_subplot(gs[1, 0])
    if reg_df is not None and len(reg_df) > 0:
        for target_fs in sampling_rates:
            sub = reg_df[reg_df["target_fs"] == target_fs].sort_values("area")
            if len(sub) == 0:
                continue
            f_det_vals = []
            for _, r in sub.iterrows():
                b1 = r.get(b1_key, r.get("b1", 0))
                sigma = r.get(sr_key, r.get("s_resid", 0))
                if b1 > 0 and sigma > 0:
                    fd = z_alpha * sigma * np.sqrt(2.0 / 30) / b1 * 100
                    f_det_vals.append(min(fd, 100))
                else:
                    f_det_vals.append(np.nan)
            ls = fs_styles.get(target_fs, "-")
            mk = fs_markers.get(target_fs, "o")
            ax.plot(sub["area"].values, f_det_vals, f"{ls}{mk}",
                    label=f"fs={target_fs} Hz", markersize=7, linewidth=2)
        ax.set_xlabel("Electrode area (cm^2)")
        ax.set_ylabel("Min detectable breach (%, N=30)")
        ax.set_title(f"Detection threshold ({y_label}, p < {p_threshold})",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
    else:
        ax.text(0.5, 0.5, "No regression data", transform=ax.transAxes,
                ha="center", fontsize=12)

    # ── Row 2 right: Sampling rate effect (fixed area, fit lines) ──
    ax = fig.add_subplot(gs[1, 1])
    ref_area = areas[len(areas) // 2] if areas else 0.2
    if raw_df is not None:
        for target_fs in sampling_rates:
            sub = raw_df[(raw_df["target_fs"] == target_fs) &
                         (raw_df["area"] == ref_area)]
            if len(sub) < 5:
                continue
            x = sub["f"].values
            y = sub[y_col].values * 1e6
            fit = _ols_fit(x, y)
            f_plot = np.linspace(0, max(x.max(), 0.5), 200)
            y_hat, ci_lo, ci_hi = _predict(fit, f_plot, kind="confidence")
            ls = fs_styles.get(target_fs, "-")
            mk = fs_markers.get(target_fs, "o")
            ax.scatter(x * 100, y, s=8, alpha=0.15, color="grey")
            ax.plot(f_plot * 100, y_hat, ls, color=None,
                    linewidth=2, label=f"fs={target_fs} Hz")
            ax.fill_between(f_plot * 100, ci_lo, ci_hi, alpha=0.10)
    else:
        for target_fs in sampling_rates:
            sub = stats_df[(stats_df["target_fs"] == target_fs) &
                           (stats_df["area"] == ref_area)].sort_values("f")
            ls = fs_styles.get(target_fs, "-")
            mk = fs_markers.get(target_fs, "o")
            ax.plot(sub["f"].values * 100, sub["rms_mean"].values * 1e6,
                    f"{ls}{mk}", label=f"fs={target_fs} Hz",
                    markersize=5, linewidth=1.5)
    ax.set_xlabel("Exposed bare metal (%)")
    ax.set_ylabel(f"{y_label} (uA)")
    ax.set_title(f"Sampling rate effect  (A = {ref_area} cm^2)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 3 left: Heatmap from REGRESSION p_slope ──
    ax = fig.add_subplot(gs[2, 0])
    if reg_df is not None and len(reg_df) > 0:
        areas_u = sorted(reg_df["area"].unique())
        fs_u = sorted(reg_df["target_fs"].unique())

        p_matrix = np.full((len(areas_u), len(fs_u)), np.nan)
        fdet_matrix = np.full((len(areas_u), len(fs_u)), np.nan)
        for i, area in enumerate(areas_u):
            for j, fs in enumerate(fs_u):
                row = reg_df[(reg_df["area"] == area) &
                             (reg_df["target_fs"] == fs)]
                if len(row):
                    pv = row[ps_key].values[0] if ps_key in row.columns else row["p_slope"].values[0]
                    p_matrix[i, j] = -np.log10(max(pv, 1e-30))
                    b1v = row[b1_key].values[0] if b1_key in row.columns else row["b1"].values[0]
                    sv = row[sr_key].values[0] if sr_key in row.columns else row["s_resid"].values[0]
                    if b1v > 0:
                        fdet_matrix[i, j] = (z_alpha * sv *
                                             np.sqrt(2.0 / 30) / b1v * 100)

        im = ax.imshow(p_matrix, aspect="auto", origin="lower",
                       cmap="RdYlGn_r", vmin=0,
                       vmax=max(15, np.nanmax(p_matrix) * 1.1))
        ax.set_xticks(range(len(fs_u)))
        ax.set_xticklabels([f"{int(fs)} Hz" for fs in fs_u], fontsize=9)
        ax.set_yticks(range(len(areas_u)))
        ax.set_yticklabels([f"{a}" for a in areas_u], fontsize=9)
        ax.set_xlabel("Sampling rate")
        ax.set_ylabel("Electrode area (cm^2)")
        ax.set_title(f"Regression significance ({y_label})", fontsize=10)
        fig.colorbar(im, ax=ax, label="-log10(p)", shrink=0.85)

        for i in range(len(areas_u)):
            for j in range(len(fs_u)):
                pv = p_matrix[i, j]
                fd = fdet_matrix[i, j]
                if not np.isnan(pv):
                    txt = ""
                    if pv > -np.log10(p_threshold):
                        txt = "*"
                    if not np.isnan(fd) and fd <= 100:
                        txt += f"\n{fd:.0f}%"
                    if txt:
                        ax.text(j, i, txt.strip(), ha="center", va="center",
                                fontsize=8, color="white", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No regression data", transform=ax.transAxes,
                ha="center", fontsize=12)

    # ── Row 3 right: R^2 and slope summary ──
    ax = fig.add_subplot(gs[2, 1])
    if reg_df is not None and len(reg_df) > 0:
        for target_fs in sampling_rates:
            sub = reg_df[reg_df["target_fs"] == target_fs].sort_values("area")
            if len(sub) == 0:
                continue
            ls = fs_styles.get(target_fs, "-")
            mk = fs_markers.get(target_fs, "o")
            r2_vals = sub[r2_key].values if r2_key in sub.columns else sub["r2"].values
            ax.plot(sub["area"].values, r2_vals, f"{ls}{mk}",
                    label=f"fs={target_fs} Hz", markersize=7, linewidth=2)
        ax.set_xlabel("Electrode area (cm^2)")
        ax.set_ylabel("R^2 (OLS fit)")
        ax.set_title(f"Regression quality ({y_label})  R^2 vs area", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No regression data", transform=ax.transAxes,
                ha="center", fontsize=12)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Summary text
# ═══════════════════════════════════════════════════════════════════════════

def _build_summary_text(reg_df, areas, sampling_rates, duration, p_threshold,
                        raw_df=None):
    """Build a monospace summary table string."""
    if reg_df is None or len(reg_df) == 0:
        return "(No regression results — need more data points per configuration.)"

    lines = []
    lines.append("=" * 110)
    lines.append("  REGRESSION ANALYSIS:  RMS = b0 + b1*f")
    lines.append(f"  Two-sample comparison (baseline vs breach), p < {p_threshold}")
    lines.append("=" * 110)

    header = (f"{'fs':>4}  {'Area':>6}  {'b1(uA/%)':>9}  "
              f"{'p(slope)':>11}  {'R^2':>6}  {'sigma(uA)':>10}  "
              f"{'N1':>5}  {'N5':>5}  {'N10':>5}  {'N30':>5}  "
              f"{'N_10%':>6}")
    lines.append("  " + header)
    lines.append("  " + "-" * 106)

    for target_fs in sampling_rates:
        for area in areas:
            row = reg_df[(reg_df["target_fs"] == target_fs) &
                         (reg_df["area"] == area)]
            if len(row) == 0:
                continue
            r = row.iloc[0]

            def _fmt_pct(val):
                return f"{val:.0f}%" if not np.isnan(val) and val <= 100 else ">100"

            lines.append(
                f"  {int(r['target_fs']):>4}  {r['area']:>6.3f}  "
                f"{r['b1_uA_per_pct']:>9.4f}  "
                f"{r['p_slope']:>11.1e}  {r['r2']:>6.3f}  "
                f"{r['s_resid']*1e6:>10.3f}  "
                f"{_fmt_pct(r['f_det_N1_pct']):>5}  "
                f"{_fmt_pct(r['f_det_N5_pct']):>5}  "
                f"{_fmt_pct(r['f_det_N10_pct']):>5}  "
                f"{_fmt_pct(r['f_det_N30_pct']):>5}  "
                f"{r['n_req_10pct']:>6.0f}"
            )
        lines.append("  " + "-" * 106)

    lines.append("")
    lines.append("  Columns  N1..N30:  min detectable breach (%) for 1,5,10,30 recordings")
    lines.append("  Column   N_10%:    recordings needed to detect 10% breach")
    lines.append("=" * 110)

    # Universal detection table
    z_alpha = stats.norm.ppf(1 - p_threshold / 2)
    reg_df2 = reg_df.copy()
    reg_df2["ratio"] = reg_df2["s_resid"] / reg_df2["b1"].replace(0, np.nan)
    ratio_mean = reg_df2["ratio"].mean()
    ratio_std = reg_df2["ratio"].std()
    if np.isnan(ratio_mean):
        ratio_mean = 0.22
        ratio_std = 0.02

    lines.append("")
    lines.append("  UNIVERSAL DETECTION TABLE")
    lines.append(f"  sigma/b1 = {ratio_mean:.3f} +/- {ratio_std:.3f}  "
                 f"(pooled over {len(reg_df)} configs)")
    lines.append(f"  {'N rec':>7}  {'Time':>7}  {'f_det':>8}")
    lines.append("  " + "-" * 30)
    for N in [1, 2, 5, 10, 20, 30, 50, 100]:
        fd = z_alpha * ratio_mean * np.sqrt(2.0 / N) * 100
        t_min = N * duration / 60
        lines.append(f"  {N:>7d}  {t_min:>5.1f} min  {fd:>7.1f} %")
    lines.append("  " + "-" * 30)
    n10 = np.ceil((z_alpha * ratio_mean * np.sqrt(2) / 0.10) ** 2)
    lines.append(f"  N for 10% detection: {n10:.0f}  "
                 f"= {n10 * duration / 60:.1f} min total recording")

    # ── Multi-metric comparison ──
    # Metrics to compare: RMS, Chebyshev k=2..5, MA(1s), Residual(1s)
    _metric_info = [
        ("RMS",       "",        "r2",  "p_slope",  "f_det_N30_pct"),
        ("Cheb2-5",   "cheb_",   "cheb_r2",  "cheb_p_slope",  "cheb_f_det_N30_pct"),
        ("MA(1s)",    "ma1s_",   "ma1s_r2",  "ma1s_p_slope",  "ma1s_f_det_N30_pct"),
        ("Resid(1s)", "resid1s_","resid1s_r2","resid1s_p_slope","resid1s_f_det_N30_pct"),
    ]
    # Check which metrics are available
    avail = [(name, pfx, r2c, pc, fdc)
             for name, pfx, r2c, pc, fdc in _metric_info
             if r2c in reg_df.columns and reg_df[r2c].notna().any()]

    if len(avail) > 1:
        lines.append("")
        lines.append("=" * 130)
        names_str = " vs ".join(m[0] for m in avail)
        lines.append(f"  METRIC COMPARISON:  {names_str}")
        lines.append("  (winner = lowest f_det at N=30 recordings)")
        lines.append("=" * 130)

        # Header
        hdr_parts = [f"{'fs':>4}  {'Area':>6}"]
        for name, *_ in avail:
            hdr_parts.append(f"{'R2_'+name:>10}  {'fdet_'+name:>10}")
        hdr_parts.append(f"{'BEST':>10}")
        lines.append("  " + "  ".join(hdr_parts))
        lines.append("  " + "-" * (len("  ".join(hdr_parts)) + 2))

        win_counter = {m[0]: 0 for m in avail}

        for _, r in reg_df.iterrows():
            parts = [f"{int(r['target_fs']):>4}  {r['area']:>6.3f}"]
            fdets = {}
            for name, pfx, r2c, pc, fdc in avail:
                r2v = r.get(r2c, np.nan)
                fdv = r.get(fdc, np.nan)
                fdets[name] = fdv

                def _fp(v):
                    return f"{v:.1f}%" if not np.isnan(v) and v <= 100 else ">100"

                parts.append(f"{r2v:>10.3f}  {_fp(fdv):>10}")

            # Find winner (lowest f_det)
            valid = {k: v for k, v in fdets.items() if not np.isnan(v)}
            if valid:
                best = min(valid, key=valid.get)
                win_counter[best] += 1
            else:
                best = "-"
            parts.append(f"{best:>10}")
            lines.append("  " + "  ".join(parts))

        lines.append("  " + "-" * 80)
        lines.append("  Win count:  " +
                     ",  ".join(f"{k}: {v}" for k, v in win_counter.items()))
        best_metric = max(win_counter, key=win_counter.get)
        lines.append(f"  >> Best overall metric: {best_metric}")

    # ── Two-factor analysis: does sampling rate significantly affect slope? ──
    if raw_df is not None and len(raw_df) > 100:
        lines.append("")
        lines.append("=" * 110)
        lines.append("  TWO-FACTOR ANALYSIS:  RMS = a0 + a1*f + a2*area + a3*fs "
                     "+ a4*(f*area) + a5*(f*fs)")
        lines.append("=" * 110)
        try:
            df2 = raw_df.copy()
            df2["rms_uA"] = df2["rms"] * 1e6
            # For each area separately, test fs effect
            for area in sorted(df2["area"].unique()):
                sub = df2[df2["area"] == area].copy()
                if len(sub) < 20:
                    continue
                x_f = sub["f"].values
                x_fs = sub["target_fs"].values
                x_interaction = x_f * x_fs
                y = sub["rms_uA"].values

                # Build design matrix: [1, f, fs, f*fs]
                X = np.column_stack([
                    np.ones(len(y)), x_f, x_fs, x_interaction
                ])
                # OLS via pseudoinverse
                beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                y_hat = X @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                n = len(y)
                p = X.shape[1]
                s2 = ss_res / max(n - p, 1)
                # Covariance of beta
                try:
                    cov_beta = s2 * np.linalg.inv(X.T @ X)
                except np.linalg.LinAlgError:
                    continue
                se_beta = np.sqrt(np.diag(cov_beta))

                # t-test for each coefficient
                t_vals = beta / np.where(se_beta > 0, se_beta, 1e-30)
                p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), max(n - p, 1)))

                r2 = 1 - ss_res / max(np.sum((y - y.mean()) ** 2), 1e-30)

                lines.append(f"  Area = {area} cm^2  (n={n}, R^2={r2:.4f})")
                labels = ["intercept", "f (breach)", "fs (rate)",
                          "f*fs (interaction)"]
                for lbl, b, se, pv in zip(labels, beta, se_beta, p_vals):
                    sig = "***" if pv < 0.001 else ("**" if pv < 0.01
                           else ("*" if pv < 0.05 else "ns"))
                    lines.append(f"    {lbl:<22s}  beta={b:>12.6f}  "
                                 f"se={se:>10.6f}  p={pv:.2e}  {sig}")
                lines.append("")
        except Exception as exc:
            lines.append(f"  (two-factor analysis error: {exc})")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# StudyWorker  (QThread)
# ═══════════════════════════════════════════════════════════════════════════

class StudyWorker(QThread):
    """Runs the Monte Carlo coating-breach study off the main thread."""

    sig_progress = pyqtSignal(int, int)       # (done, total)
    sig_status = pyqtSignal(str)              # status message
    sig_finished = pyqtSignal(object, object, object)  # raw_df, reg_df, stats_df

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        cfg = self.cfg
        areas = cfg["areas"]
        fractions = cfg["fractions"]
        sampling_rates = cfg["sampling_rates"]
        n_mc = cfg["n_mc"]
        duration = cfg["duration"]

        total = len(areas) * len(fractions) * len(sampling_rates) * n_mc
        self.sig_status.emit(f"Preparing {total} simulations...")

        # Set module-level config for worker processes
        global _STUDY_CFG
        _STUDY_CFG = cfg

        # Build task list
        tasks = []
        for area in areas:
            for target_fs in sampling_rates:
                for f in fractions:
                    for i in range(n_mc):
                        seed = abs(hash((area, target_fs, f, i))) % (2 ** 31)
                        tasks.append((f, area, target_fs, seed))

        t0 = time.time()
        n_workers = min(mp.cpu_count(), 8)
        results = []

        try:
            self.sig_status.emit(f"Launching {n_workers} workers...")
            with mp.Pool(n_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                for idx, r in enumerate(
                    pool.imap_unordered(_run_single_sim, tasks, chunksize=10)
                ):
                    if self._cancelled:
                        pool.terminate()
                        self.sig_status.emit("Cancelled.")
                        return
                    results.append(r)
                    done = idx + 1
                    if done % max(1, total // 100) == 0 or done == total:
                        self.sig_progress.emit(done, total)
                        elapsed = time.time() - t0
                        rate = done / max(elapsed, 0.01)
                        eta = (total - done) / max(rate, 0.01)
                        self.sig_status.emit(
                            f"[{done}/{total}]  {elapsed:.0f}s elapsed,  ETA {eta:.0f}s"
                        )
        except Exception:
            # Fallback to sequential
            self.sig_status.emit("Multiprocessing failed, running sequentially...")
            results = []
            for idx, task in enumerate(tasks):
                if self._cancelled:
                    self.sig_status.emit("Cancelled.")
                    return
                results.append(_run_single_sim(task))
                done = idx + 1
                if done % max(1, total // 50) == 0 or done == total:
                    self.sig_progress.emit(done, total)

        elapsed = time.time() - t0
        self.sig_status.emit(f"Done: {total} sims in {elapsed:.1f}s ({elapsed/60:.1f} min). Analyzing...")

        raw_df = pd.DataFrame(results)

        # Statistics
        stats_df = _compute_statistics(raw_df, areas, fractions, sampling_rates,
                                       cfg["p_threshold"])
        # Regression
        reg_df = _regression_analysis(raw_df, areas, sampling_rates,
                                      cfg["p_threshold"])

        self.sig_finished.emit(raw_df, reg_df, stats_df)


def _init_worker(cfg):
    """Initializer for pool workers — sets module-level config."""
    global _STUDY_CFG
    _STUDY_CFG = cfg


# ═══════════════════════════════════════════════════════════════════════════
# StudyTab  (main widget)
# ═══════════════════════════════════════════════════════════════════════════

class StudyTab(QWidget):
    """Interactive coating breach detection study tab.

    Physical / instrument parameters are read from the main window.
    The coating preset (with film) is selected here or taken from the
    main page.  Breach = film removal, same interface underneath.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.mw = main_window          # reference to MainWindow
        self._worker = None
        self._raw_df = None
        self._reg_df = None
        self._stats_df = None
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

        # --- Metal Interface (bare metal, same under coating) ---
        grp_interface = QGroupBox("Metal Interface (same everywhere)")
        iface_layout = QFormLayout()

        self.combo_interface = QComboBox()
        # Show all presets; user selects the metal-electrolyte interface
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
        self.combo_interface.setToolTip(
            "Metal-electrolyte interface (Rct, CPEdl).\n"
            "This is the SAME for coated and bare zones —\n"
            "the coating doesn't change the underlying metal."
        )
        iface_layout.addRow("Metal preset:", self.combo_interface)

        self.chk_use_main_page = QCheckBox("Use interface from main page")
        self.chk_use_main_page.setChecked(False)
        self.chk_use_main_page.setToolTip(
            "If checked, interface parameters (Rct, CPEdl)\n"
            "are taken from the Simulation page."
        )
        self.chk_use_main_page.stateChanged.connect(self._on_coated_source_changed)
        iface_layout.addRow(self.chk_use_main_page)

        grp_interface.setLayout(iface_layout)
        left_layout.addWidget(grp_interface)

        # --- Coating Film (removed on breach) ---
        grp_film = QGroupBox("Protective Film (removed on breach)")
        film_layout = QFormLayout()

        self.spin_rf = QDoubleSpinBox()
        self.spin_rf.setRange(1.0, 100000.0)
        self.spin_rf.setValue(362.0)
        self.spin_rf.setSuffix("  Ohm*cm^2")
        self.spin_rf.setDecimals(1)
        self.spin_rf.setSingleStep(10.0)
        self.spin_rf.setToolTip(
            "Film resistance (specific).\n"
            "Default 362 = Rp_coated(662) - Rct_bare(300)\n"
            "for chitosan on AZ91."
        )
        film_layout.addRow("Rf:", self.spin_rf)

        self.spin_y0f = QDoubleSpinBox()
        self.spin_y0f.setRange(0.1, 50000.0)
        self.spin_y0f.setValue(800.0)
        self.spin_y0f.setSuffix("  uF/cm^2")
        self.spin_y0f.setDecimals(1)
        self.spin_y0f.setSingleStep(10.0)
        self.spin_y0f.setToolTip("Film CPE admittance Y0.")
        film_layout.addRow("Y0_f:", self.spin_y0f)

        self.spin_nf = QDoubleSpinBox()
        self.spin_nf.setRange(0.01, 1.0)
        self.spin_nf.setValue(0.70)
        self.spin_nf.setDecimals(2)
        self.spin_nf.setSingleStep(0.05)
        self.spin_nf.setToolTip("Film CPE exponent (0..1).")
        film_layout.addRow("n_f:", self.spin_nf)

        self.lbl_model_info = QLabel()
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setStyleSheet(
            "color: #555; font-size: 9px; font-style: italic; padding: 2px;"
        )
        self.lbl_model_info.setText(
            "Coated = Film(Rf,CPEf) + Interface(Rct,CPEdl)\n"
            "Breach = film removed, same interface exposed\n"
            "Rf should capture total coating protection"
        )
        film_layout.addRow(self.lbl_model_info)

        grp_film.setLayout(film_layout)
        left_layout.addWidget(grp_film)

        # Connect combo/spin changes to live snapshot refresh
        self.combo_interface.currentIndexChanged.connect(self._on_circuit_combo_changed)
        self.spin_rf.valueChanged.connect(self._on_circuit_combo_changed)
        self.spin_y0f.valueChanged.connect(self._on_circuit_combo_changed)
        self.spin_nf.valueChanged.connect(self._on_circuit_combo_changed)

        # --- Snapshot: parameters from main page ---
        grp_snapshot = QGroupBox("Physical Parameters (from main page)")
        snap_layout = QVBoxLayout()
        self.lbl_snapshot = QLabel("(click Run to capture)")
        self.lbl_snapshot.setWordWrap(True)
        self.lbl_snapshot.setStyleSheet(
            "color: #444; font-size: 10px; background: #f8f8f0; "
            "border: 1px solid #ccc; border-radius: 4px; padding: 6px;"
        )
        snap_layout.addWidget(self.lbl_snapshot)
        grp_snapshot.setLayout(snap_layout)
        left_layout.addWidget(grp_snapshot)

        # --- Experimental Design ---
        grp_design = QGroupBox("Experimental Design")
        design_layout = QFormLayout()

        self.edit_areas = QLineEdit("0.05, 0.1, 0.2, 0.5, 1.0")
        self.edit_areas.setToolTip("Comma-separated electrode areas in cm^2")
        design_layout.addRow("Areas (cm^2):", self.edit_areas)

        self.edit_fractions = QLineEdit("0, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50")
        self.edit_fractions.setToolTip("Comma-separated exposed fractions in %")
        design_layout.addRow("Fractions (%):", self.edit_fractions)

        self.spin_n_mc = QSpinBox()
        self.spin_n_mc.setRange(5, 200)
        self.spin_n_mc.setValue(30)
        self.spin_n_mc.setToolTip("Number of Monte Carlo realisations per point")
        design_layout.addRow("N_MC:", self.spin_n_mc)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(1.0, 300.0)
        self.spin_duration.setValue(30.0)
        self.spin_duration.setSuffix(" s")
        self.spin_duration.setToolTip("Duration of each simulated recording")
        design_layout.addRow("Duration:", self.spin_duration)

        self.chk_fs_1 = QCheckBox("1 Hz")
        self.chk_fs_1.setChecked(True)
        self.chk_fs_10 = QCheckBox("10 Hz")
        self.chk_fs_10.setChecked(True)
        self.chk_fs_100 = QCheckBox("100 Hz")
        self.chk_fs_100.setChecked(True)
        fs_row = QHBoxLayout()
        fs_row.addWidget(self.chk_fs_1)
        fs_row.addWidget(self.chk_fs_10)
        fs_row.addWidget(self.chk_fs_100)
        design_layout.addRow("Sampling rates:", fs_row)

        self.spin_p_threshold = QDoubleSpinBox()
        self.spin_p_threshold.setRange(0.001, 0.1)
        self.spin_p_threshold.setValue(0.01)
        self.spin_p_threshold.setDecimals(3)
        self.spin_p_threshold.setSingleStep(0.005)
        self.spin_p_threshold.setToolTip("Statistical significance threshold")
        design_layout.addRow("p threshold:", self.spin_p_threshold)

        grp_design.setLayout(design_layout)
        left_layout.addWidget(grp_design)

        # --- Run / Cancel / Progress ---
        self.btn_run = QPushButton("Run Study")
        self.btn_run.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px;"
        )
        self.btn_run.clicked.connect(self._on_run)
        left_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(
            "background-color: #f44336; color: white; padding: 6px;"
        )
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

        # --- Import / Export ---
        grp_io = QGroupBox("Import / Export")
        io_layout = QVBoxLayout()

        self.btn_import_csv = QPushButton("Load Previous Study...")
        self.btn_import_csv.setToolTip(
            "Load raw + regression CSVs from a previous run.\n"
            "Expected files: *_raw.csv and *_reg.csv"
        )
        self.btn_import_csv.clicked.connect(self._on_import_csv)
        io_layout.addWidget(self.btn_import_csv)

        export_row = QHBoxLayout()
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        export_row.addWidget(self.btn_export_csv)

        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_png.setEnabled(False)
        self.btn_export_png.clicked.connect(self._on_export_png)
        export_row.addWidget(self.btn_export_png)

        io_layout.addLayout(export_row)
        grp_io.setLayout(io_layout)
        left_layout.addWidget(grp_io)

        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll)

        # ── Right panel: results ──────────────────────────────────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.result_tabs = QTabWidget()

        self.canvas_regression = FigureCanvas(Figure(figsize=(5, 4)))
        self.result_tabs.addTab(self.canvas_regression, "Regression")

        self.canvas_areas = FigureCanvas(Figure(figsize=(5, 4)))
        self.result_tabs.addTab(self.canvas_areas, "Area Comparison")

        self.canvas_pointwise = FigureCanvas(Figure(figsize=(5, 4)))
        self.result_tabs.addTab(self.canvas_pointwise, "Point-wise")

        right_splitter.addWidget(self.result_tabs)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setFontFamily("Consolas")
        self.txt_summary.setFontPointSize(9)
        self.txt_summary.setMinimumHeight(120)
        self.txt_summary.setPlaceholderText("Run a study to see results here...")
        right_splitter.addWidget(self.txt_summary)

        right_splitter.setStretchFactor(0, 7)
        right_splitter.setStretchFactor(1, 3)

        main_layout.addWidget(right_splitter, stretch=1)

        # Initial snapshot
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._refresh_snapshot)

    # ── Helpers ────────────────────────────────────────────────────

    def _parse_float_list(self, text: str) -> list:
        """Parse comma-separated float list from a QLineEdit."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        result = []
        for p in parts:
            try:
                result.append(float(p))
            except ValueError:
                pass
        return sorted(set(result))

    def _on_coated_source_changed(self):
        """Toggle controls based on 'use main page' checkbox."""
        use_main = self.chk_use_main_page.isChecked()
        self.combo_interface.setEnabled(not use_main)

        # Also disable film spinboxes if main page has film
        main_has_film = False
        if use_main and self.mw is not None:
            topo = self.mw.circuit_builder.get_topology()
            main_has_film = topo.get("has_film", False)
        film_editable = not (use_main and main_has_film)
        self.spin_rf.setEnabled(film_editable)
        self.spin_y0f.setEnabled(film_editable)
        self.spin_nf.setEnabled(film_editable)

        self._refresh_snapshot()

    def _on_circuit_combo_changed(self):
        self._refresh_snapshot()

    def _refresh_snapshot(self):
        """Refresh the read-only snapshot label with current selections."""
        try:
            mw_params = self._read_main_window_params()
            if not mw_params:
                mw_params = {
                    "r_shunt": 10.0, "rho_tissue": 100.0,
                    "electrode_dist": 0.5, "r_contact": 10.0,
                    "internal_fs": 10000, "adc_lsb": 0.0,
                    "amplifier_noise": 0.0, "ecg_enabled": False,
                    "ecg_rate": 400.0, "ecg_amp": 1.5e-3,
                }

            if self.chk_use_main_page.isChecked() and self.mw is not None:
                cb = self.mw.circuit_builder
                all_sp = cb.get_specific_params()
                topo = cb.get_topology()
                interface_specific = {
                    k: v for k, v in all_sp.items()
                    if k in ("Rct", "Y0_dl", "n_dl", "sigma_w", "L", "RL")
                }
                interface_name = f"{cb.combo_preset.currentText()} (main page)"
                if topo.get("has_film", False):
                    film_specific = {
                        "Rf":   all_sp.get("Rf", self.spin_rf.value()),
                        "Y0_f": all_sp.get("Y0_f", self.spin_y0f.value()),
                        "n_f":  all_sp.get("n_f", self.spin_nf.value()),
                    }
                else:
                    film_specific = {
                        "Rf":   self.spin_rf.value(),
                        "Y0_f": self.spin_y0f.value(),
                        "n_f":  self.spin_nf.value(),
                    }
            else:
                iface_key = self.combo_interface.currentData()
                iface_preset = PRESETS.get(iface_key, PRESETS["bare_early"])
                active = get_active_param_names(
                    iface_preset.has_film, iface_preset.has_warburg,
                    iface_preset.has_inductive,
                )
                interface_specific = {k: iface_preset.defaults[k]
                                      for k in active}
                interface_name = iface_preset.name

                film_specific = {
                    "Rf":   self.spin_rf.value(),
                    "Y0_f": self.spin_y0f.value(),
                    "n_f":  self.spin_nf.value(),
                }

            self._update_snapshot_label(mw_params, interface_name,
                                        interface_specific, film_specific)
        except Exception:
            pass  # Don't crash on preview update

    def _get_sampling_rates(self) -> list:
        rates = []
        if self.chk_fs_1.isChecked():
            rates.append(1)
        if self.chk_fs_10.isChecked():
            rates.append(10)
        if self.chk_fs_100.isChecked():
            rates.append(100)
        return rates if rates else [10]

    def _read_main_window_params(self) -> dict:
        """Read physical / instrument parameters from the main window widgets.

        NOTE: coated circuit is determined by _build_cfg (from study tab
        preset or main page, depending on checkbox).
        """
        mw = self.mw
        if mw is None:
            return {}

        r_contact = 10.0
        rho = mw.spin_rho.value()
        area_main = mw.spin_area.value()
        rs_auto = mw.spin_rs.value()
        electrode_dist = max((rs_auto - r_contact) * area_main / max(rho, 0.01), 0.01)

        # Electrode positions for dipole model (compositor uses these)
        try:
            electrode_pos = mw.electrode_widget.get_positions()
        except Exception:
            electrode_pos = {"working": [0.48, 0.65], "counter": [0.52, 0.65]}

        return {
            "r_shunt": mw.spin_rshunt.value(),
            "rho_tissue": rho,
            "electrode_dist": electrode_dist,
            "r_contact": 10.0,
            "internal_fs": mw.spin_internal_fs.value(),
            "adc_lsb": mw.spin_adc_lsb.value() * 1e-6,   # uA -> A
            "amplifier_noise": mw.spin_white_noise.value() * 1e-6,  # uA -> A
            "ecg_enabled": mw.chk_ecg.isChecked(),
            "ecg_rate": mw.spin_hr.value(),
            "ecg_amp": mw.spin_ecg_amp.value() * 1e-3,    # mV -> V
            "electrode_pos": electrode_pos,
        }

    def _update_snapshot_label(self, mw_params: dict, interface_name: str,
                               interface_specific: dict, film_specific: dict):
        """Show a compact read-only summary of study circuit + physical params."""
        p = mw_params

        rct = interface_specific.get("Rct", 0)
        rf = film_specific.get("Rf", 0)
        y0_f = film_specific.get("Y0_f", 0)
        n_f = film_specific.get("n_f", 0)

        total_rp = rct + rf  # approximate DC polarization resistance

        lines = [
            f"Interface: {interface_name}",
            f"  Rct = {rct:.0f} Ohm*cm^2",
            f"Film: Rf = {rf:.1f},  Y0_f = {y0_f:.0f},  n_f = {n_f:.2f}",
            f"Total Rp (coated) ~ {total_rp:.0f} Ohm*cm^2",
            f"  (Rf/Rct = {rf/max(rct,0.1):.1%} of interface)",
        ]

        lines += [
            "",
            f"R_shunt = {p.get('r_shunt', 0):.1f} Ohm",
            f"rho = {p.get('rho_tissue', 0):.0f} Ohm*cm,  "
            f"dist = {p.get('electrode_dist', 0):.2f} cm",
            f"ADC LSB = {p.get('adc_lsb', 0)*1e6:.3f} uA,  "
            f"Amp noise = {p.get('amplifier_noise', 0)*1e6:.3f} uA",
            f"ECG: {'ON' if p.get('ecg_enabled') else 'OFF'}  "
            f"({p.get('ecg_rate', 0):.0f} BPM, "
            f"{p.get('ecg_amp', 0)*1e3:.1f} mV)",
        ]
        self.lbl_snapshot.setText("\n".join(lines))

    def _build_cfg(self) -> dict:
        """Collect all parameters into a config dict."""
        areas = self._parse_float_list(self.edit_areas.text())
        fracs_pct = self._parse_float_list(self.edit_fractions.text())
        fractions = [f / 100.0 for f in fracs_pct]

        # Read physical/instrument parameters from the main window
        mw_params = self._read_main_window_params()

        # Fallback defaults if main window not connected
        if not mw_params:
            mw_params = {
                "r_shunt": 10.0,
                "rho_tissue": 100.0,
                "electrode_dist": 0.5,
                "r_contact": 10.0,
                "internal_fs": 10000,
                "adc_lsb": 0.05e-6,
                "amplifier_noise": 0.05e-6,
                "ecg_enabled": True,
                "ecg_rate": 400.0,
                "ecg_amp": 1.5e-3,
            }

        # --- Metal interface + film ---
        if self.chk_use_main_page.isChecked() and self.mw is not None:
            cb = self.mw.circuit_builder
            all_sp = cb.get_specific_params()
            topo = cb.get_topology()
            # Extract interface params
            interface_specific = {
                k: v for k, v in all_sp.items()
                if k in ("Rct", "Y0_dl", "n_dl", "sigma_w", "L", "RL")
            }
            # Extract film params from main page too (if film enabled)
            if topo.get("has_film", False):
                film_specific = {
                    "Rf":   all_sp.get("Rf", self.spin_rf.value()),
                    "Y0_f": all_sp.get("Y0_f", self.spin_y0f.value()),
                    "n_f":  all_sp.get("n_f", self.spin_nf.value()),
                }
            else:
                # Main page has no film — use study tab spinboxes
                film_specific = {
                    "Rf":   self.spin_rf.value(),
                    "Y0_f": self.spin_y0f.value(),
                    "n_f":  self.spin_nf.value(),
                }
            interface_name = f"{cb.combo_preset.currentText()} (main page)"
            has_warburg = topo.get("has_warburg", False)
            has_inductive = topo.get("has_inductive", False)
        else:
            iface_key = self.combo_interface.currentData()
            iface_preset = PRESETS.get(iface_key, PRESETS["bare_early"])
            active = get_active_param_names(
                iface_preset.has_film, iface_preset.has_warburg,
                iface_preset.has_inductive,
            )
            interface_specific = {k: iface_preset.defaults[k] for k in active}
            interface_name = iface_preset.name
            has_warburg = iface_preset.has_warburg
            has_inductive = iface_preset.has_inductive

            # Film parameters from study tab spinboxes
            film_specific = {
                "Rf":   self.spin_rf.value(),
                "Y0_f": self.spin_y0f.value(),
                "n_f":  self.spin_nf.value(),
            }

        # Build the full "coated" specific params = interface + film
        coated_specific = dict(interface_specific)
        coated_specific.update(film_specific)

        coated_topo = {
            "has_film": True,  # film always present for breach study
            "has_warburg": has_warburg,
            "has_inductive": has_inductive,
        }

        self._update_snapshot_label(mw_params, interface_name,
                                    interface_specific, film_specific)

        # Merge everything
        cfg = dict(mw_params)
        cfg.update({
            "coated_specific": coated_specific,
            "coated_topo": coated_topo,
            "coated_preset_name": interface_name,
            "areas": areas,
            "fractions": fractions,
            "sampling_rates": self._get_sampling_rates(),
            "n_mc": self.spin_n_mc.value(),
            "duration": self.spin_duration.value(),
            "p_threshold": self.spin_p_threshold.value(),
        })
        return cfg

    # ── Run / Cancel ──────────────────────────────────────────────

    def _on_run(self):
        cfg = self._build_cfg()

        if not cfg["areas"]:
            QMessageBox.warning(self, "Input Error", "No valid areas specified.")
            return
        if not cfg["fractions"]:
            QMessageBox.warning(self, "Input Error", "No valid fractions specified.")
            return

        # Validate: Rf should be positive for meaningful breach detection
        rf = cfg["coated_specific"].get("Rf", 0)
        rct = cfg["coated_specific"].get("Rct", 0)
        if rf <= 0:
            QMessageBox.warning(
                self, "No Film Protection",
                "Film resistance Rf is zero or negative.\n\n"
                "The breach model removes the film from a fraction of\n"
                "the surface. Without Rf, there is nothing to detect.\n\n"
                "Set Rf > 0 in the 'Protective Film' section.",
            )
            return
        if rf < rct * 0.05:
            reply = QMessageBox.warning(
                self, "Very Low Film Protection",
                f"Rf = {rf:.1f} Ohm*cm^2 is only {rf/rct:.1%} of Rct = {rct:.0f}.\n\n"
                f"The impedance change from film removal will be very small\n"
                f"and may not be detectable in the noise.\n\n"
                f"For chitosan on AZ91, Rf ~ 362 Ohm*cm^2 is typical.\n\n"
                f"Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_png.setEnabled(False)
        self.progress_bar.setValue(0)

        self._worker = StudyWorker(cfg, parent=self)
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

    def _on_finished(self, raw_df, reg_df, stats_df):
        self._raw_df = raw_df
        self._reg_df = reg_df
        self._stats_df = stats_df

        cfg = self._worker.cfg

        # Build figures
        self.lbl_status.setText("Generating plots...")

        try:
            self._show_regression(raw_df, reg_df, cfg)
        except Exception as e:
            self.lbl_status.setText(f"Regression plot error: {e}")

        try:
            self._show_areas(raw_df, reg_df, cfg)
        except Exception as e:
            self.lbl_status.setText(f"Area plot error: {e}")

        try:
            self._show_pointwise(stats_df, cfg, raw_df=raw_df, reg_df=reg_df)
        except Exception as e:
            self.lbl_status.setText(f"Point-wise plot error: {e}")

        # Summary text
        try:
            summary = _build_summary_text(
                reg_df, cfg["areas"], cfg["sampling_rates"],
                cfg["duration"], cfg["p_threshold"],
                raw_df=raw_df,
            )
            self.txt_summary.setPlainText(summary)
        except Exception as e:
            self.txt_summary.setPlainText(f"Error building summary: {e}")

        self.btn_export_csv.setEnabled(True)
        self.btn_export_png.setEnabled(True)
        self.lbl_status.setText("Study complete.")

    def _on_worker_done(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    # ── Plot display ──────────────────────────────────────────────

    def _show_regression(self, raw_df, reg_df, cfg):
        fig = _make_regression_figure(
            raw_df, reg_df, cfg["areas"], cfg["sampling_rates"],
            cfg["duration"], cfg["amplifier_noise"], cfg["adc_lsb"],
            cfg["p_threshold"],
        )
        self._replace_canvas(self.result_tabs, 0, fig, "Regression")

    def _show_areas(self, raw_df, reg_df, cfg):
        # Use the middle sampling rate for area comparison
        fs_list = cfg["sampling_rates"]
        target_fs = fs_list[len(fs_list) // 2] if fs_list else 10
        fig = _make_area_comparison_figure(
            raw_df, reg_df, cfg["areas"], target_fs,
            cfg["duration"], cfg["amplifier_noise"], cfg["adc_lsb"],
            cfg["n_mc"], cfg["p_threshold"],
        )
        self._replace_canvas(self.result_tabs, 1, fig, "Area Comparison")

    def _show_pointwise(self, stats_df, cfg, raw_df=None, reg_df=None):
        fig = _make_pointwise_figure(
            stats_df, cfg["areas"], cfg["fractions"],
            cfg["sampling_rates"], cfg["duration"],
            cfg["amplifier_noise"], cfg["adc_lsb"],
            cfg["p_threshold"],
            raw_df=raw_df, reg_df=reg_df,
        )
        self._replace_canvas(self.result_tabs, 2, fig, "Point-wise")

    def _replace_canvas(self, tabs: QTabWidget, index: int,
                        fig: Figure, title: str):
        """Replace a canvas at the given tab index with a new Figure."""
        old = tabs.widget(index)
        new_canvas = FigureCanvas(fig)
        new_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        tabs.removeTab(index)
        tabs.insertTab(index, new_canvas, title)
        tabs.setCurrentIndex(index)
        if old is not None:
            old.deleteLater()
        # Store references for export
        if index == 0:
            self.canvas_regression = new_canvas
        elif index == 1:
            self.canvas_areas = new_canvas
        elif index == 2:
            self.canvas_pointwise = new_canvas

    # ── Import ─────────────────────────────────────────────────────

    def _on_import_csv(self):
        """Load raw + reg CSVs from a previous study run."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Study CSV (raw)",
            "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            raw_df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Import Error",
                                 f"Cannot read CSV:\n{e}")
            return

        # Validate required columns
        required = {"rms", "f", "area", "target_fs"}
        if not required.issubset(raw_df.columns):
            QMessageBox.critical(
                self, "Import Error",
                f"CSV is missing columns: {required - set(raw_df.columns)}\n"
                f"Found: {list(raw_df.columns)}"
            )
            return

        self._raw_df = raw_df

        # Try loading companion *_reg.csv
        base = os.path.splitext(path)[0]
        # Strip _raw suffix if present
        if base.endswith("_raw"):
            base = base[:-4]
        reg_path = base + "_reg.csv"
        stats_path = base + "_stats.csv"

        # Derive areas & sampling rates from raw data
        areas = sorted(raw_df["area"].unique().tolist())
        sampling_rates = sorted(
            int(x) for x in raw_df["target_fs"].unique().tolist()
        )
        fractions = sorted(raw_df["f"].unique().tolist())

        # Attempt to load reg CSV or recompute
        if os.path.isfile(reg_path):
            try:
                self._reg_df = pd.read_csv(reg_path)
            except Exception:
                self._reg_df = _regression_analysis(
                    raw_df, areas, sampling_rates,
                    self.spin_p_threshold.value())
        else:
            self._reg_df = _regression_analysis(
                raw_df, areas, sampling_rates,
                self.spin_p_threshold.value())

        # Attempt to load stats CSV or recompute
        if os.path.isfile(stats_path):
            try:
                self._stats_df = pd.read_csv(stats_path)
            except Exception:
                self._stats_df = _compute_statistics(
                    raw_df, fractions, areas, sampling_rates,
                    self.spin_p_threshold.value())
        else:
            self._stats_df = _compute_statistics(
                raw_df, fractions, areas, sampling_rates,
                self.spin_p_threshold.value())

        # Build a synthetic cfg for the plotting functions
        p_threshold = self.spin_p_threshold.value()
        duration = self.spin_duration.value()

        cfg = {
            "areas": areas,
            "fractions": fractions,
            "sampling_rates": sampling_rates,
            "duration": duration,
            "p_threshold": p_threshold,
            "amplifier_noise": 0.0,
            "adc_lsb": 0.0,
            "n_mc": int(raw_df.groupby(
                ["f", "area", "target_fs"]).size().median())
            if len(raw_df) > 0 else 30,
        }

        # Re-plot
        self.lbl_status.setText("Loading imported data...")
        try:
            self._show_regression(raw_df, self._reg_df, cfg)
        except Exception as e:
            self.lbl_status.setText(f"Regression plot error: {e}")
        try:
            self._show_areas(raw_df, self._reg_df, cfg)
        except Exception as e:
            self.lbl_status.setText(f"Area plot error: {e}")
        try:
            self._show_pointwise(self._stats_df, cfg,
                                 raw_df=raw_df, reg_df=self._reg_df)
        except Exception as e:
            self.lbl_status.setText(f"Point-wise plot error: {e}")
        try:
            summary = _build_summary_text(
                self._reg_df, areas, sampling_rates, duration, p_threshold,
                raw_df=raw_df)
            self.txt_summary.setPlainText(summary)
        except Exception as e:
            self.txt_summary.setPlainText(f"Error: {e}")

        self.btn_export_csv.setEnabled(True)
        self.btn_export_png.setEnabled(True)
        n_rows = len(raw_df)
        self.lbl_status.setText(
            f"Loaded {n_rows} rows from {os.path.basename(path)}")

    # ── Export ─────────────────────────────────────────────────────

    def _on_export_csv(self):
        if self._raw_df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Raw CSV", "study_coating_breach_raw.csv",
            "CSV Files (*.csv)"
        )
        if path:
            self._raw_df.to_csv(path, index=False)
            # Also save regression CSV alongside
            base = os.path.splitext(path)[0]
            if self._reg_df is not None:
                self._reg_df.to_csv(base + "_reg.csv", index=False)
            if self._stats_df is not None:
                self._stats_df.to_csv(base + "_stats.csv", index=False)
            self.lbl_status.setText(f"Exported to {path}")

    def _on_export_png(self):
        if self._raw_df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figures", "study_coating_breach_results.png",
            "PNG Files (*.png)"
        )
        if path:
            base = os.path.splitext(path)[0]
            # Save all three canvases
            for canvas, suffix in [
                (self.canvas_regression, "_regression"),
                (self.canvas_areas, "_areas"),
                (self.canvas_pointwise, "_pointwise"),
            ]:
                if canvas and canvas.figure:
                    out = base + suffix + ".png"
                    canvas.figure.savefig(out, dpi=200, bbox_inches="tight")
            self.lbl_status.setText(f"Exported PNGs to {base}_*.png")
