"""
Composable impedance element library for EIS equivalent circuit modeling.

Elements: Z_R, Z_C, Z_CPE, Z_L, Z_W, Series, Parallel

Usage:
    # Randles circuit: Rs - (CPE || Rct)
    circuit = Series(Z_R(Rs), Parallel(Z_CPE(Y0, n), Z_R(Rct)))
    Z = circuit.impedance(omega)

Literature basis: AZ91/AZ91D in SBF/Hank's/Ringer
  - Gerengi 2022 (ACS Omega), Khalili & Tamjid 2021 (Sci Rep),
  - Jamshidi 2025 (J Mater Sci), Fekry & El-Sherif 2009 (Electrochim. Acta)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# Core impedance elements
# ═══════════════════════════════════════════════════════════════

class ImpedanceElement(ABC):
    """Base class for all impedance elements."""

    @abstractmethod
    def impedance(self, omega: np.ndarray) -> np.ndarray:
        """Return complex impedance Z(omega) for angular frequency array."""
        pass


class Z_R(ImpedanceElement):
    """Pure resistance: Z = R."""
    def __init__(self, R: float):
        self.R = R

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        return np.full_like(omega, self.R, dtype=complex)


class Z_C(ImpedanceElement):
    """Ideal capacitor: Z = 1 / (j*omega*C)."""
    def __init__(self, C: float):
        self.C = C

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Z = np.empty_like(omega, dtype=complex)
        nz = omega != 0
        Z[nz] = 1.0 / (1j * omega[nz] * self.C)
        Z[~nz] = 1e30  # open circuit at DC
        return Z


class Z_CPE(ImpedanceElement):
    """
    Constant Phase Element: Z = 1 / (Y0 * (j*omega)^n).

    n = 1.0  ->  ideal capacitor
    n = 0.5  ->  Warburg-like
    n = 0.0  ->  pure resistor

    Parameters
    ----------
    Y0 : float   CPE admittance prefactor [S * s^n]
    n  : float   CPE exponent (0 < n <= 1)
    """
    def __init__(self, Y0: float, n: float):
        self.Y0 = max(Y0, 1e-30)
        self.n = np.clip(n, 0.01, 1.0)

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Z = np.empty_like(omega, dtype=complex)
        nz = omega != 0
        Z[nz] = 1.0 / (self.Y0 * (1j * omega[nz]) ** self.n)
        Z[~nz] = 1e30
        return Z

    def effective_capacitance(self, R_parallel: float) -> float:
        """Brug formula (1984): C_eff = (Y0 * R^(1-n))^(1/n)."""
        if self.n <= 0 or self.n > 1 or R_parallel <= 0:
            return self.Y0
        return (self.Y0 * R_parallel ** (1.0 - self.n)) ** (1.0 / self.n)


class Z_L(ImpedanceElement):
    """Inductor: Z = j*omega*L."""
    def __init__(self, L: float):
        self.L = L

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        return 1j * omega * self.L


class Z_W(ImpedanceElement):
    """Semi-infinite Warburg: Z = sigma_w / sqrt(j*omega)."""
    def __init__(self, sigma_w: float):
        self.sigma_w = sigma_w

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Z = np.empty_like(omega, dtype=complex)
        nz = omega != 0
        Z[nz] = self.sigma_w / np.sqrt(1j * omega[nz])
        Z[~nz] = 1e30
        return Z


class Series(ImpedanceElement):
    """Series connection: Z = Z1 + Z2 + ..."""
    def __init__(self, *elements: ImpedanceElement):
        self.elements = list(elements)

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Z = np.zeros_like(omega, dtype=complex)
        for e in self.elements:
            Z += e.impedance(omega)
        return Z


class Parallel(ImpedanceElement):
    """Parallel connection: 1/Z = 1/Z1 + 1/Z2 + ..."""
    def __init__(self, *elements: ImpedanceElement):
        self.elements = list(elements)

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Y = np.zeros_like(omega, dtype=complex)
        has_short = np.zeros(omega.shape, dtype=bool)

        for e in self.elements:
            Z_e = e.impedance(omega)
            abs_Z = np.abs(Z_e)
            # short circuit (Z ≈ 0): marks the parallel result as 0
            short = abs_Z < 1e-30
            has_short |= short
            # open circuit (Z very large): contributes nothing
            open_c = abs_Z >= 1e29
            # normal: add admittance
            valid = (~short) & (~open_c)
            Y[valid] += 1.0 / Z_e[valid]

        # Safe division: substitute Y=0 with a dummy before dividing
        # (np.where evaluates both branches, so bare 1/Y would warn on zeros)
        nonzero = np.abs(Y) > 1e-30
        safe_Y = np.where(nonzero, Y, 1.0 + 0j)
        result = np.where(nonzero, 1.0 / safe_Y, 1e30 + 0j)
        result[has_short] = 0.0 + 0j   # short wins in parallel
        return result


# ═══════════════════════════════════════════════════════════════
# Circuit builder
# ═══════════════════════════════════════════════════════════════

def build_electrode_circuit(
    params: Dict[str, float],
    has_film: bool = False,
    has_warburg: bool = False,
    has_inductive: bool = False,
) -> Tuple[ImpedanceElement, float, float]:
    """
    Build electrode-interface equivalent circuit from **absolute** parameters.

    Topology
    --------
    [Film?] - [Interface] - [Inductive?]

    Film:       Parallel(CPE_f, R_f)
    Interface:  Parallel(CPE_dl, R_ct)           or
                Parallel(CPE_dl, Series(R_ct, W))
    Inductive:  Parallel(L, R_L)

    Returns
    -------
    (circuit, R_ct, C_dl_effective)
    """
    blocks: list = []

    # 1. Film / coating layer ------------------------------------------------
    if has_film:
        blocks.append(
            Parallel(Z_CPE(params["Y0_f"], params["n_f"]), Z_R(params["Rf"]))
        )

    # 2. Charge-transfer interface -------------------------------------------
    R_ct = params["Rct"]
    cpe_dl = Z_CPE(params["Y0_dl"], params["n_dl"])

    if has_warburg:
        r_branch = Series(Z_R(R_ct), Z_W(params["sigma_w"]))
    else:
        r_branch = Z_R(R_ct)

    blocks.append(Parallel(cpe_dl, r_branch))

    # 3. Inductive loop ------------------------------------------------------
    if has_inductive:
        blocks.append(Parallel(Z_L(params["L"]), Z_R(params["RL"])))

    # Assemble ---------------------------------------------------------------
    circuit = Series(*blocks) if len(blocks) > 1 else blocks[0]
    C_dl_eff = cpe_dl.effective_capacitance(R_ct)

    return circuit, R_ct, C_dl_eff


# ═══════════════════════════════════════════════════════════════
# Presets  (all values are *specific*, i.e. per cm²)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PresetDef:
    name: str
    description: str
    has_film: bool
    has_warburg: bool
    has_inductive: bool
    defaults: Dict[str, float]


PRESETS = {
    "bare_early": PresetDef(
        "Bare AZ91 (early stage)",
        "Single time constant, weak product film.\n"
        
        "Rs–(CPEdl‖Rct), Rp ≈ 291 Ω·cm², icorr ≈ 34 μA/cm².",
        False, False, False,
        {"Rct": 300.0, "Y0_dl": 25.0, "n_dl": 0.85},
    ),
    "bare_film": PresetDef(
        "Bare AZ91 (developed film)",
        "Two time constants, Mg(OH)₂/phosphate film.\n"
        
        "Rs–(CPEf‖Rf)–(CPEdl‖Rct), Rp ≈ 500 Ω·cm².",
        True, False, False,
        {
            "Rf": 10.0, "Y0_f": 800.0, "n_f": 0.70,
            "Rct": 500.0, "Y0_dl": 25.0, "n_dl": 0.85,
        },
    ),
    "coated_chitosan": PresetDef(
        "Coated (chitosan)",
        "Chitosan coating on AZ91.\n"
        
        "Rs–(CPEcoat‖Rcoat)–(CPEdl‖Rct), Rp ≈ 662 Ω·cm².",
        True, False, False,
        {
            "Rf": 10.6, "Y0_f": 800.0, "n_f": 0.70,
            "Rct": 652.0, "Y0_dl": 30.0, "n_dl": 0.85,
        },
    ),
    "coated_mof": PresetDef(
        "Coated (chitosan + MOF)",
        "Chitosan/ZIF-8 on AZ91.\n"
        
        "Rs–(CPEdl‖Rct), Rp ≈ 1256 Ω·cm², icorr ≈ 6.5 μA/cm².",
        False, False, False,
        {"Rct": 1256.0, "Y0_dl": 20.0, "n_dl": 0.90},
    ),
    "bare_inductive": PresetDef(
        "Bare AZ91 (inductive loop)",
        "Three time constants with adsorption loop.\n"
        
        "Rs–(CPEf‖Rf)–(CPEdl‖Rct)–(L‖RL).",
        True, False, True,
        {
            "Rf": 50.0, "Y0_f": 200.0, "n_f": 0.75,
            "Rct": 1100.0, "Y0_dl": 30.0, "n_dl": 0.80,
            "L": 5.0, "RL": 500.0,
        },
    ),
}


# ═══════════════════════════════════════════════════════════════
# Parameter metadata (for UI generation)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParamMeta:
    display: str        # UI label
    unit: str           # suffix
    min_val: float
    max_val: float
    default: float
    decimals: int = 2
    step: float = 1.0
    conv: str = "R"     # "R" -> R_abs = R_sp / A
                        # "C" -> Y_abs = Y_sp * A * 1e-6  (μF -> F)
                        # "none" -> no conversion


PARAM_META = {
    # --- interface (always present) ---
    "Rct":     ParamMeta("Rct",   "Ω·cm²",    1.0,  100000.0,  300.0, 1, 10.0, "R"),
    "Y0_dl":   ParamMeta("Y₀ dl", "μF/cm²",   0.1,  10000.0,   25.0, 1,  1.0, "C"),
    "n_dl":    ParamMeta("n dl",  "",          0.01,      1.0,   0.85, 2, 0.05, "none"),
    # --- film / coating ---
    "Rf":      ParamMeta("Rf",    "Ω·cm²",    0.1,  100000.0,   10.0, 1,  1.0, "R"),
    "Y0_f":    ParamMeta("Y₀ f",  "μF/cm²",   0.1,  10000.0,  800.0, 1, 10.0, "C"),
    "n_f":     ParamMeta("n f",   "",          0.01,      1.0,   0.70, 2, 0.05, "none"),
    # --- warburg ---
    "sigma_w": ParamMeta("σ_w",   "Ω·cm²/√s", 0.1,  100000.0,  100.0, 1, 10.0, "R"),
    # --- inductive ---
    "L":       ParamMeta("L",     "H·cm²",    0.001, 1000.0,     5.0, 3,  0.1, "R"),
    "RL":      ParamMeta("RL",    "Ω·cm²",    0.1,  100000.0,  500.0, 1, 10.0, "R"),
}


def get_active_param_names(
    has_film: bool, has_warburg: bool, has_inductive: bool
) -> List[str]:
    """Ordered parameter names for current topology."""
    names: list = []
    if has_film:
        names += ["Rf", "Y0_f", "n_f"]
    names += ["Rct", "Y0_dl", "n_dl"]
    if has_warburg:
        names += ["sigma_w"]
    if has_inductive:
        names += ["L", "RL"]
    return names


def convert_to_absolute(params_specific: Dict[str, float], area_cm2: float) -> Dict[str, float]:
    """
    Convert specific (per cm²) parameter values to absolute.

    R_abs = R_sp / area          (Ω)
    Y_abs = Y_sp * area * 1e-6   (F, from μF/cm²)
    n     = unchanged
    L_abs = L_sp / area          (H)
    σ_abs = σ_sp / area          (Ω/√s)
    """
    if area_cm2 < 1e-9:
        area_cm2 = 1e-9
    result: dict = {}
    for key, val in params_specific.items():
        meta = PARAM_META.get(key)
        if meta is None or meta.conv == "none":
            result[key] = val
        elif meta.conv == "R":
            result[key] = val / area_cm2
        elif meta.conv == "C":
            result[key] = val * area_cm2 * 1e-6   # μF -> F
    return result
