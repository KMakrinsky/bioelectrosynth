"""
Visual circuit builder widget for the BioElectroSynth UI.

Contains:
  - CircuitSchematicWidget  — QPainter-based live schematic
  - CircuitBuilderWidget    — preset selector + schematic + block toggles + params
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QCheckBox, QDoubleSpinBox, QLabel,
    QFrame, QSizePolicy, QToolTip,
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont,
    QPainterPath, QFontMetrics,
)

from .circuit_model import (
    PRESETS, PARAM_META, get_active_param_names,
    convert_to_absolute, build_electrode_circuit,
)


# ═══════════════════════════════════════════════════════════════
# Element colour palette
# ═══════════════════════════════════════════════════════════════

_ELEM_STYLES = {
    #           fill                    border                  text
    "R":   (QColor(200, 222, 255),  QColor(55, 120, 220),  QColor(25, 65, 140)),
    "Q":   (QColor(200, 240, 210),  QColor(40, 160, 70),   QColor(15, 95, 35)),
    "L":   (QColor(255, 215, 215),  QColor(220, 60, 55),   QColor(140, 25, 25)),
    "W":   (QColor(255, 235, 200),  QColor(230, 170, 20),  QColor(140, 95, 0)),
    "auto": (QColor(228, 228, 228), QColor(150, 150, 150), QColor(100, 100, 100)),
}

_WIRE_COLOR = QColor(75, 75, 75)
_BG_COLOR = QColor(248, 249, 250)
_NODE_COLOR = QColor(55, 55, 55)

_LABEL_FONT = QFont("Segoe UI", 8)
_LABEL_FONT.setBold(True)
_BLOCK_LABEL_FONT = QFont("Segoe UI", 7)
_BLOCK_LABEL_FONT.setItalic(True)


# ═══════════════════════════════════════════════════════════════
# Schematic widget
# ═══════════════════════════════════════════════════════════════

class CircuitSchematicWidget(QWidget):
    """Draws the electrode equivalent-circuit schematic."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(125)
        self.setMaximumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.has_film = False
        self.has_warburg = False
        self.has_inductive = False

    # -- public API ----------------------------------------------------------

    def set_topology(self, has_film: bool, has_warburg: bool, has_inductive: bool):
        self.has_film = has_film
        self.has_warburg = has_warburg
        self.has_inductive = has_inductive
        self.update()

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W = self.width()
        H = self.height()

        # background
        p.fillRect(0, 0, W, H, _BG_COLOR)

        # vertical geometry
        y_mid = H // 2
        y_top = H // 4 + 2
        y_bot = 3 * H // 4 - 2
        eh = 22                 # element box height

        # build block descriptors
        blocks = self._blocks()
        n_blk = len(blocks)

        # horizontal geometry
        margin = 12
        rs_w = 34
        gap = 14               # wire gap between blocks
        avail = W - 2 * margin - rs_w - gap * (n_blk + 1)
        blk_w = max(65, min(120, avail // max(n_blk, 1)))
        total = rs_w + gap + n_blk * blk_w + (n_blk - 1) * gap
        x = (W - total) // 2

        wire = QPen(_WIRE_COLOR, 2)

        # ── left terminal ──
        p.setPen(wire)
        self._dot(p, x, y_mid)
        x += 4

        # ── Rs ──
        self._elem_box(p, x, y_mid - eh // 2, rs_w, eh, "Rs", "auto")
        x += rs_w
        p.setPen(wire)
        p.drawLine(int(x), y_mid, int(x + gap), y_mid)
        x += gap

        # ── parallel blocks ──
        for i, blk in enumerate(blocks):
            x_end = x + blk_w

            # vertical split
            p.setPen(wire)
            p.drawLine(int(x), y_mid, int(x), y_top)
            p.drawLine(int(x), y_mid, int(x), y_bot)
            self._dot(p, x, y_mid)

            pad = 4
            inner = blk_w - 2 * pad

            # ── top branch ──
            tops = blk["top"]   # list of (label, type)
            n_top = len(tops)
            ew = max(20, (inner - 4 * (n_top - 1)) // n_top)
            tx = x + pad
            for j, (lbl, etype) in enumerate(tops):
                p.setPen(wire)
                p.drawLine(int(x if j == 0 else tx), y_top, int(tx), y_top)
                self._elem_box(p, tx, y_top - eh // 2, ew, eh, lbl, etype)
                tx += ew
                if j < n_top - 1:
                    p.setPen(wire)
                    p.drawLine(int(tx), y_top, int(tx + 4), y_top)
                    tx += 4
            p.setPen(wire)
            p.drawLine(int(tx), y_top, int(x_end), y_top)

            # ── bottom branch ──
            bot_lbl, bot_type = blk["bot"]
            bw = inner
            bx = x + pad
            p.setPen(wire)
            p.drawLine(int(x), y_bot, int(bx), y_bot)
            self._elem_box(p, bx, y_bot - eh // 2, bw, eh, bot_lbl, bot_type)
            p.setPen(wire)
            p.drawLine(int(bx + bw), y_bot, int(x_end), y_bot)

            # rejoin
            p.setPen(wire)
            p.drawLine(int(x_end), y_top, int(x_end), y_mid)
            p.drawLine(int(x_end), y_bot, int(x_end), y_mid)
            self._dot(p, x_end, y_mid)

            # block label (above)
            p.setFont(_BLOCK_LABEL_FONT)
            p.setPen(QPen(QColor(120, 120, 120)))
            lbl_rect = QRectF(x, 1, float(blk_w), float(y_top - eh // 2 - 3))
            p.drawText(lbl_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                       blk.get("label", ""))

            x = x_end
            if i < n_blk - 1:
                p.setPen(wire)
                p.drawLine(int(x), y_mid, int(x + gap), y_mid)
                x += gap

        # ── right terminal ──
        p.setPen(wire)
        p.drawLine(int(x), y_mid, int(x + 4), y_mid)
        x += 4
        self._dot(p, x, y_mid)

        p.end()

    # -- helpers -------------------------------------------------------------

    def _blocks(self):
        """Return list of block descriptors for current topology."""
        blks = []
        if self.has_film:
            blks.append({
                "label": "Film",
                "top": [("Rf", "R")],
                "bot": ("Qf", "Q"),
            })
        if self.has_warburg:
            blks.append({
                "label": "Interface",
                "top": [("Rct", "R"), ("W", "W")],
                "bot": ("Qdl", "Q"),
            })
        else:
            blks.append({
                "label": "Interface",
                "top": [("Rct", "R")],
                "bot": ("Qdl", "Q"),
            })
        if self.has_inductive:
            blks.append({
                "label": "Inductive",
                "top": [("RL", "R")],
                "bot": ("L", "L"),
            })
        return blks

    def _dot(self, p: QPainter, x, y, r=3):
        p.setPen(QPen(_NODE_COLOR, 1))
        p.setBrush(QBrush(_NODE_COLOR))
        p.drawEllipse(QPointF(float(x), float(y)), r, r)

    def _elem_box(self, p: QPainter, x, y, w, h, label, etype):
        """Draw a rounded-rect element with colour-coded accent."""
        fill, border, text = _ELEM_STYLES.get(etype, _ELEM_STYLES["auto"])
        rect = QRectF(float(x), float(y), float(w), float(h))

        # box
        p.setPen(QPen(border, 1.5))
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(rect, 5, 5)

        # accent bar on the left
        accent_w = 4
        accent_rect = QRectF(float(x), float(y), accent_w, float(h))
        accent_path = QPainterPath()
        accent_path.addRoundedRect(accent_rect, 5, 5)
        # clip to left half only
        clip = QRectF(float(x), float(y), accent_w + 1, float(h))
        accent_path2 = QPainterPath()
        accent_path2.addRect(clip)
        final_accent = accent_path & accent_path2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(border))
        p.drawPath(final_accent)

        # label
        p.setFont(_LABEL_FONT)
        p.setPen(QPen(text))
        text_rect = QRectF(float(x + accent_w + 1), float(y), float(w - accent_w - 2), float(h))
        p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)


# ═══════════════════════════════════════════════════════════════
# Circuit builder panel
# ═══════════════════════════════════════════════════════════════

class CircuitBuilderWidget(QWidget):
    """
    Complete circuit builder:  preset + schematic + toggles + parameter spinboxes.

    Signals
    -------
    sigCircuitChanged   emitted whenever topology or parameter values change.
    """

    sigCircuitChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._applying_preset = False
        self._param_spins: dict = {}
        self._param_labels: dict = {}
        self._setup_ui()

    # ── UI setup ───────────────────────────────────────────────

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # --- preset ---
        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        for key, preset in PRESETS.items():
            self.combo_preset.addItem(preset.name, key)
        self.combo_preset.addItem("Custom", "custom")
        self.combo_preset.setToolTip("")  # set dynamically
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        row_preset.addWidget(self.combo_preset, 1)
        lay.addLayout(row_preset)

        # --- schematic ---
        self.schematic = CircuitSchematicWidget()
        lay.addWidget(self.schematic)

        # --- block toggles ---
        row_tog = QHBoxLayout()
        self.chk_film = QCheckBox("Film/Coat")
        self.chk_film.setToolTip("Rf ‖ CPEf  — film or coating layer")
        self.chk_film.stateChanged.connect(self._on_toggle)

        self.chk_warburg = QCheckBox("Warburg")
        self.chk_warburg.setToolTip("W (semi-∞ diffusion) inside interface block")
        self.chk_warburg.stateChanged.connect(self._on_toggle)

        self.chk_inductive = QCheckBox("Inductive")
        self.chk_inductive.setToolTip("L ‖ RL  — adsorption / film breakdown loop")
        self.chk_inductive.stateChanged.connect(self._on_toggle)

        row_tog.addWidget(self.chk_film)
        row_tog.addWidget(self.chk_warburg)
        row_tog.addWidget(self.chk_inductive)
        lay.addLayout(row_tog)

        # --- separator ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(line)

        # --- description ---
        self.lbl_desc = QLabel("")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color: #666; font-size: 10px;")
        lay.addWidget(self.lbl_desc)

        # --- parameter form ---
        self._params_form = QFormLayout()
        self._params_form.setContentsMargins(0, 0, 0, 0)
        self._create_all_param_spinboxes()
        lay.addLayout(self._params_form)

        # apply default
        self._apply_preset("bare_early")

    def _create_all_param_spinboxes(self):
        """Pre-create spinboxes for every possible parameter."""
        # Desired display order: film params, then interface, then warburg, then inductive
        ordered_keys = ["Rf", "Y0_f", "n_f", "Rct", "Y0_dl", "n_dl", "sigma_w", "L", "RL"]
        for key in ordered_keys:
            meta = PARAM_META[key]
            spin = QDoubleSpinBox()
            spin.setRange(meta.min_val, meta.max_val)
            spin.setValue(meta.default)
            spin.setDecimals(meta.decimals)
            spin.setSingleStep(meta.step)
            if meta.unit:
                spin.setSuffix(f"  {meta.unit}")
            spin.valueChanged.connect(self._on_param_changed)

            lbl = QLabel(f"{meta.display}:")
            self._param_spins[key] = spin
            self._param_labels[key] = lbl
            self._params_form.addRow(lbl, spin)

    # ── preset / toggle logic ──────────────────────────────────

    def _apply_preset(self, preset_key: str):
        self._applying_preset = True

        if preset_key != "custom":
            preset = PRESETS.get(preset_key)
            if preset:
                self.chk_film.setChecked(preset.has_film)
                self.chk_warburg.setChecked(preset.has_warburg)
                self.chk_inductive.setChecked(preset.has_inductive)

                for key, val in preset.defaults.items():
                    if key in self._param_spins:
                        self._param_spins[key].setValue(val)

                self.lbl_desc.setText(preset.description)
                self.combo_preset.setToolTip(preset.description)

        self._sync_ui()
        self._applying_preset = False
        self.sigCircuitChanged.emit()

    def _on_preset_changed(self, _index):
        key = self.combo_preset.currentData()
        self._apply_preset(key)

    def _on_toggle(self):
        if self._applying_preset:
            return
        # auto-switch to Custom
        if self.combo_preset.currentData() != "custom":
            self._applying_preset = True
            idx = self.combo_preset.findData("custom")
            self.combo_preset.setCurrentIndex(idx)
            self._applying_preset = False
            self.lbl_desc.setText("Custom topology.")

        self._sync_ui()
        self.sigCircuitChanged.emit()

    def _on_param_changed(self):
        if not self._applying_preset:
            self.sigCircuitChanged.emit()

    def _sync_ui(self):
        """Update param visibility and schematic to match current toggles."""
        hf = self.chk_film.isChecked()
        hw = self.chk_warburg.isChecked()
        hi = self.chk_inductive.isChecked()

        active = set(get_active_param_names(hf, hw, hi))
        for key in PARAM_META:
            vis = key in active
            self._param_spins[key].setVisible(vis)
            self._param_labels[key].setVisible(vis)

        self.schematic.set_topology(hf, hw, hi)

    # ── public API ─────────────────────────────────────────────

    def get_specific_params(self) -> dict:
        """Current parameter values in specific (per cm²) units."""
        hf = self.chk_film.isChecked()
        hw = self.chk_warburg.isChecked()
        hi = self.chk_inductive.isChecked()
        active = get_active_param_names(hf, hw, hi)
        return {k: self._param_spins[k].value() for k in active}

    def get_circuit(self, area_cm2: float):
        """
        Build electrode circuit with absolute values.

        Returns
        -------
        (circuit: ImpedanceElement, R_ct_abs: float, C_dl_eff: float)
        """
        specific = self.get_specific_params()
        absolute = convert_to_absolute(specific, area_cm2)
        return build_electrode_circuit(
            absolute,
            has_film=self.chk_film.isChecked(),
            has_warburg=self.chk_warburg.isChecked(),
            has_inductive=self.chk_inductive.isChecked(),
        )

    def get_topology(self) -> dict:
        return {
            "has_film": self.chk_film.isChecked(),
            "has_warburg": self.chk_warburg.isChecked(),
            "has_inductive": self.chk_inductive.isChecked(),
        }

    def set_enabled_all(self, enabled: bool):
        """Enable / disable all controls (used when impedance model is off)."""
        self.combo_preset.setEnabled(enabled)
        self.chk_film.setEnabled(enabled)
        self.chk_warburg.setEnabled(enabled)
        self.chk_inductive.setEnabled(enabled)
        for spin in self._param_spins.values():
            spin.setEnabled(enabled)
