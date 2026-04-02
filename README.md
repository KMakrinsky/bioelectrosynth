# BioElectroSynth

A physics-based digital twin of an implantable zero-resistance ammeter (ZRA) corrosion sensor operating inside a murine model. The simulator combines a two-zone equivalent circuit (intact coating + exposed metal) with biopotential interference (ECG, EMG), thermal noise, and amplifier noise to generate realistic synthetic ZRA signals. It provides a graphical interface for parameter exploration, Monte Carlo coating-breach detection studies, and electrode-area/SNR optimisation.

## Features

- **Signal decomposition** — real-time visualisation of corrosion current, ECG/EMG interference, thermal and amplifier noise as separate channels.
- **Two-zone impedance model** — Randles circuit with CPE, solution resistance, and configurable coating film parameters.
- **Monte Carlo breach detection** — sweep electrode area, breach fraction, and sampling rate; compare RMS, residual, moving-average, and Chebyshev spectroscopy metrics.
- **SNR phase diagram** — automatic electrode-area optimisation with 2D color-mapped SNR landscape.
- **3D murine phantom** — OpenGL volume-conductor model with draggable electrodes and real-time geometric gain calculation.
- **Export** — save simulation data as CSV/NPZ, export figures as PNG.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone https://github.com/<your-username>/bioelectrosynth.git
cd bioelectrosynth
pip install -r requirements.txt
```

## Quick Start

```bash
python main_3d.py     # 3D murine phantom mode
```

## Project Structure

```
bioelectrosynth/
├── main_3d.py               # 3D entry point (OpenGL phantom)
├── requirements.txt
├── src/
│   ├── ui.py                # Main window, parameter controls, simulation tabs
│   ├── circuit_model.py     # Two-zone Randles impedance model (CPE, Warburg)
│   ├── circuit_widget.py    # Interactive circuit schematic widget
│   ├── physics_engine.py    # ECG/EMG biopotential generation, thermal noise
│   ├── compositor.py        # Signal compositor: combines corrosion + bio + noise
│   ├── mouse_phantom.py     # 2D electrode placement on mouse silhouette
│   ├── widget_3d.py         # 3D OpenGL murine phantom viewer
│   ├── study_tab.py         # Monte Carlo coating-breach study tab
│   └── optimization_tab.py  # Electrode area / SNR optimisation tab
└── tests/
    └── test_cheb_spectroscopy.py  # Unit tests for Chebyshev noise metrics
```

## Citation

If you use BioElectroSynth in your research, please cite the underlying study:

```bibtex
@article{makrinsky2026invivo,
  author  = {Makrinsky Kirill, Klyuev, Alexey, Batishchev Oleg},
  title   = {Modelling of {In} {Vivo} Electrochemical Noise: A Computational Framework to Optimize the Corrosion Monitoring of Biodegradable Magnesium Implants},
  journal = {Journal of Functional Biomaterials},
  year    = {2026},
  publisher = {MDPI},
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
