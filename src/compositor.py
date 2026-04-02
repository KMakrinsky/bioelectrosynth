import numpy as np
import scipy.signal as signal
import pandas as pd
from typing import Dict, Any, Tuple
from .physics_engine import CorrosionEngine, BioEngine, SensorEngine

class SignalCompositor:
    """
    Orchestrates the generation of signals from different engines and combines them.
    """
    
    def __init__(self):
        pass
        
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the composite signal based on parameters.
        
        Args:
            params: Dictionary containing configuration for all engines.
                    Structure:
                    {
                        "duration": float,
                        "fs": int,
                        "corrosion": { ... kwargs for CorrosionEngine ... },
                        "bio": { ... kwargs for BioEngine ... },
                        "sensor": { ... kwargs for SensorEngine ... }
                    }
                    
        Returns:
            Dictionary with keys:
            - 'total_signal': np.ndarray
            - 'time': np.ndarray
            - 'corrosion_component': np.ndarray
            - 'bio_component': np.ndarray
            - 'sensor_component': np.ndarray
            - 'mask_bubbles': np.ndarray
            - 'mask_emg': np.ndarray
            - ... other individual components
        """
        duration = params.get("duration", 10.0)
        target_fs = params.get("fs", 1000)
        # Physics simulation rate (High frequency)
        internal_fs = params.get("internal_fs", 10000)
        
        # Ensure internal_fs is at least target_fs
        if internal_fs < target_fs:
            internal_fs = target_fs
            
        # Instantiate engines
        corrosion_params = params.get("corrosion", {})
        bio_params = params.get("bio", {})
        sensor_params = params.get("sensor", {})
        
        # Calculate electrode ECG pickup from coordinates (if present).
        # Skip if the caller already provides a custom lead_gain (e.g. 3D phantom).
        # Uses a cardiac-dipole volume-conductor model:
        #   V_diff ∝ (d / r²) · cos(θ)
        # where d = inter-electrode spacing, r = distance from heart to midpoint,
        # θ = angle between electrode vector and heart dipole axis (approx. cranio-caudal).
        # Reference: Malmivuo & Plonsey, "Bioelectromagnetism", Ch. 11.
        electrodes = params.get("electrodes", {})
        if electrodes and not bio_params.pop("skip_dipole_model", False):
            # Heart position in normalized [0,1]×[0,1] coordinates
            heart_pos = np.array([0.5, 0.5])
            # Approximate cardiac dipole axis (cranio-caudal, slightly left)
            dipole_axis = np.array([0.0, -1.0])  # pointing head→tail in our coords
            dipole_axis = dipole_axis / np.linalg.norm(dipole_axis)

            pos_working = np.array(electrodes.get("working", [0.4, 0.4]))
            pos_counter = np.array(electrodes.get("counter", [0.6, 0.6]))

            midpoint = (pos_working + pos_counter) / 2.0
            r_vec = midpoint - heart_pos
            r = max(np.linalg.norm(r_vec), 0.05)  # distance heart→midpoint

            # Inter-electrode vector and spacing
            d_vec = pos_working - pos_counter
            d = max(np.linalg.norm(d_vec), 0.01)  # electrode separation

            # Cosine between electrode pair vector and dipole axis
            # (differential pickup is maximised when electrodes are aligned with dipole)
            cos_theta = abs(np.dot(d_vec / d, dipole_axis))
            cos_theta = max(cos_theta, 0.05)  # minimum coupling

            # Differential dipole gain: ∝ d · cos(θ) / r²
            # Normalised so that d=0.4, r=0.15, cos=1 gives gain ≈ 1.0
            # (electrodes spanning chest at heart level ≈ maximum Lead-II-like pickup)
            ref_gain = 0.4 / (0.15 ** 2)  # ≈ 17.78 — normalisation constant
            dipole_gain = (d * cos_theta / (r ** 2)) / ref_gain

            bio_params["lead_gain"] = bio_params.get("lead_gain", 1.0) * dipole_gain

        corrosion_engine = CorrosionEngine(**corrosion_params)
        bio_engine = BioEngine(**bio_params)
        sensor_engine = SensorEngine(**sensor_params)
        
        # --- Signal Mixing & Impedance ---
        #
        # Physical model  (ZRA = zero-resistance ammeter):
        #
        #   Corrosion noise = CURRENT source I_n(ω) in parallel with Z_electrode
        #     → Faradaic fluctuations at the metal/electrolyte interface
        #     → Double-layer capacitance C_dl shunts high-frequency components
        #     → Measured shunt current:
        #         I_shunt(ω) = I_n(ω) · H(ω)
        #         H(ω) = Z_electrode(ω) / Z_loop(ω)   ← current-divider
        #
        #   ECG/EMG   = VOLTAGE source V_ECG (Volts) — cardiac dipole
        #     → I_ECG(ω) = V_ECG(ω) / Z_loop(ω)
        #
        #   Sensor    = CURRENT noise (amplifier, mains)
        #     → added directly to measured current
        #
        #   Thermal   = Johnson-Nyquist VOLTAGE noise V_th, PSD = 4kT·Re[Z]
        #     → I_th(ω) = V_th(ω) / Z_loop(ω)
        #
        # 1. Generate raw sources at high resolution
        i_corr_hr, comps_corr_hr = corrosion_engine.generate_current(duration, internal_fs)
        v_bio_hr, comps_bio_hr   = bio_engine.generate(duration, internal_fs)  # Voltage!
        v_sensor_hr, comps_sensor_hr = sensor_engine.generate(duration, internal_fs)

        use_circuit = corrosion_params.get("use_circuit_model", False)

        # 2. When impedance model ON, apply frequency-domain transfer functions
        if use_circuit:
            n_hr = len(v_bio_hr)
            freqs_hr = np.fft.rfftfreq(n_hr, d=1.0 / internal_fs)
            omega_hr = 2.0 * np.pi * freqs_hr

            # Compute Z_electrode and Z_loop
            if corrosion_engine.electrode_circuit is not None:
                Z_elec = corrosion_engine.electrode_circuit.impedance(omega_hr)
            else:
                jw_rc = 1.0 + 1j * omega_hr * corrosion_engine.R_ct * corrosion_engine.C_dl
                Z_elec = corrosion_engine.R_ct / jw_rc

            if corrosion_engine.use_symmetric_electrodes:
                Z_loop = corrosion_engine.R_s + 2.0 * Z_elec + corrosion_engine.R_shunt
            else:
                Z_loop = corrosion_engine.R_s + Z_elec + corrosion_engine.R_shunt

            # Safe division (avoid /0 at DC for inductors etc.)
            abs_Z = np.abs(Z_loop)
            safe_Z = np.where(abs_Z > 1e-10, Z_loop, 1e10 + 0j)

            # ── 2a. Corrosion noise: I_source → I_shunt via current divider ──
            #    H(ω) = Z_electrode(ω) / Z_loop(ω)
            #    At DC:  H ≈ R_ct / (R_s + [1|2]·R_ct + R_shunt)
            #    At HF:  C_dl shorts R_ct → Z_elec→0 → H→0  (low-pass!)
            H_corr = Z_elec / safe_Z
            I_corr_fft = np.fft.rfft(i_corr_hr)
            I_corr_filtered_fft = I_corr_fft * H_corr
            i_corr_filtered = np.fft.irfft(I_corr_filtered_fft, n=n_hr)

            # ── 2b. V_ECG → I_ECG ────────────────────────────────────────────
            V_bio_fft = np.fft.rfft(v_bio_hr)
            I_bio_fft = V_bio_fft / safe_Z
            i_bio_hr = np.fft.irfft(I_bio_fft, n=n_hr)
        else:
            # Legacy / direct mode: no impedance filtering
            i_corr_filtered = i_corr_hr
            i_bio_hr = v_bio_hr

        # 3. Mix currents (all in Amps)
        total_current_hr = i_corr_filtered + i_bio_hr

        # 4. Assemble final ZRA current
        if use_circuit:
            # ── 4a. Thermal noise: V_th → I_th = V_th / Z_loop ──────────────
            v_thermal_hr = corrosion_engine.generate_thermal_noise(n_hr, internal_fs)
            V_th_fft = np.fft.rfft(v_thermal_hr)
            I_th_fft = V_th_fft / safe_Z
            i_thermal_hr = np.fft.irfft(I_th_fft, n=n_hr)

            # Total ZRA current (all in Amps)
            total_signal_hr = total_current_hr + i_thermal_hr + v_sensor_hr

            # Individual components for layer visualisation (current)
            sig_corr_hr   = i_corr_filtered   # impedance-filtered corrosion
            sig_bio_hr    = i_bio_hr           # I_ECG = V_ECG / Z
            sig_sensor_hr = v_sensor_hr        # amplifier noise (current units)

            comps_corr_hr["thermal_noise_current"] = i_thermal_hr
            comps_corr_hr["corrosion_source_unfiltered"] = i_corr_hr
            comps_corr_hr["transfer_function_dc"] = float(np.real(H_corr[0]))
        else:
            # Direct mode: everything is just "Signal" (no impedance transform)
            total_signal_hr = total_current_hr + v_sensor_hr

            sig_corr_hr   = i_corr_hr
            sig_bio_hr    = v_bio_hr
            sig_sensor_hr = v_sensor_hr
            
        t_hr = np.arange(len(total_signal_hr)) / internal_fs
        
        # --- Downsampling to Target FS ---
        num_samples_target = int(duration * target_fs)
        ratio = internal_fs / target_fs
        # 'ideal' — Sigma-Delta decimation (Sinc-like);  'naive' — slicing (shows aliasing)
        adc_type = params.get("adc_type", "ideal")
        
        if ratio == 1.0:
            total_signal_sampled = total_signal_hr
            sig_corr = sig_corr_hr
            sig_bio = sig_bio_hr
            sig_sensor = sig_sensor_hr
            # Masks can be just passed
            comps_corr = comps_corr_hr
            comps_bio = comps_bio_hr
            comps_sensor = comps_sensor_hr
        else:
            if adc_type == "naive":
                # Naive decimation (slicing) - Demonstrates ALIASING
                # Taking every Nth sample
                step = int(ratio)
                if step < 1: step = 1
                
                total_signal_sampled = total_signal_hr[::step][:num_samples_target]
                sig_corr = sig_corr_hr[::step][:num_samples_target]
                sig_bio = sig_bio_hr[::step][:num_samples_target]
                sig_sensor = sig_sensor_hr[::step][:num_samples_target]
                
                # Masks are also sliced
                comps_corr = {k: v[::step][:num_samples_target] for k,v in comps_corr_hr.items()}
                comps_bio = {k: v[::step][:num_samples_target] for k,v in comps_bio_hr.items()}
                comps_sensor = {k: v[::step][:num_samples_target] for k,v in comps_sensor_hr.items()}
                
            elif adc_type == "sigma_delta":
                # Sigma-Delta simulation (decimate uses lowpass filter before downsampling)
                # This approximates the Sinc filter of AD7176-2
                # scipy.signal.decimate uses Chebyshev Type I by default or FIR.
                # For Sinc approximation, decimate is a good "good ADC" proxy.
                
                q = int(ratio)
                if q < 1: q = 1
                
                # Helper: resample array, skip scalars
                def _resample_comp(v, n_target, use_decimate=False, q_val=1):
                    if not isinstance(v, np.ndarray) or v.ndim == 0:
                        return v   # scalar metadata — pass through
                    if "mask" in str(id(v)):  # can't check by key here
                        pass
                    if use_decimate:
                        return signal.decimate(v, q_val)
                    return signal.resample(v, n_target)

                def _resample_dict(d, n_target, use_decimate=False, q_val=1):
                    out = {}
                    for k, v in d.items():
                        if not isinstance(v, np.ndarray) or v.ndim == 0:
                            out[k] = v
                        elif "mask" in k:
                            res = signal.resample(v, n_target)
                            out[k] = (res > 0.1).astype(float)
                        elif use_decimate:
                            out[k] = signal.decimate(v, q_val)
                        else:
                            out[k] = signal.resample(v, n_target)
                    return out

                # Check if we can use decimate (q must be integer > 1)
                if abs(ratio - q) < 1e-5 and q > 1:
                    total_signal_sampled = signal.decimate(total_signal_hr, q)
                    sig_corr = signal.decimate(sig_corr_hr, q)
                    sig_bio = signal.decimate(sig_bio_hr, q)
                    sig_sensor = signal.decimate(sig_sensor_hr, q)

                    comps_corr = _resample_dict(comps_corr_hr, num_samples_target, True, q)
                    comps_bio = _resample_dict(comps_bio_hr, num_samples_target, True, q)
                    comps_sensor = _resample_dict(comps_sensor_hr, num_samples_target, True, q)
                else:
                    total_signal_sampled = signal.resample(total_signal_hr, num_samples_target)
                    sig_corr = signal.resample(sig_corr_hr, num_samples_target)
                    sig_bio = signal.resample(sig_bio_hr, num_samples_target)
                    sig_sensor = signal.resample(sig_sensor_hr, num_samples_target)

                    comps_corr = _resample_dict(comps_corr_hr, num_samples_target)
                    comps_bio = _resample_dict(comps_bio_hr, num_samples_target)
                    comps_sensor = _resample_dict(comps_sensor_hr, num_samples_target)

            else:
                # Default: Ideal Fourier Resample (prev behavior)
                total_signal_sampled = signal.resample(total_signal_hr, num_samples_target)
                sig_corr = signal.resample(sig_corr_hr, num_samples_target)
                sig_bio = signal.resample(sig_bio_hr, num_samples_target)
                sig_sensor = signal.resample(sig_sensor_hr, num_samples_target)

                def _resample_dict_default(d, n_target):
                    out = {}
                    for k, v in d.items():
                        if not isinstance(v, np.ndarray) or v.ndim == 0:
                            out[k] = v
                        elif "mask" in k:
                            res = signal.resample(v, n_target)
                            out[k] = (res > 0.1).astype(float)
                        else:
                            out[k] = signal.resample(v, n_target)
                    return out

                comps_corr = _resample_dict_default(comps_corr_hr, num_samples_target)
                comps_bio = _resample_dict_default(comps_bio_hr, num_samples_target)
                comps_sensor = _resample_dict_default(comps_sensor_hr, num_samples_target)

        # --- ADC Amplitude Quantization (LSB) ---
        adc_lsb = params.get("adc_lsb", 0.0)
        if adc_lsb > 0:
            # Quantize the total sampled signal to the nearest LSB step
            # This simulates finite bit-depth of a real ADC (e.g., AD7176-2)
            total_signal_sampled = np.round(total_signal_sampled / adc_lsb) * adc_lsb

        # Time vector for sampled
        t_sampled = np.arange(len(total_signal_sampled)) / target_fs
        
        # Result dictionary
        result = {
            "time": t_sampled,
            "total_signal": total_signal_sampled,
            
            # High Res Data for Visualization
            "time_high_res": t_hr,
            "signal_high_res": total_signal_hr,
            "corrosion_high_res": sig_corr_hr,
            "bio_high_res": sig_bio_hr,
            "sensor_high_res": sig_sensor_hr,
            
            # Main components (sampled)
            "corrosion_component": sig_corr,
            "bio_component": sig_bio,
            "sensor_component": sig_sensor,
            
            # Detailed components/masks (flattening the dicts)
            **comps_corr,
            **comps_bio,
            **comps_sensor
        }
        
        return result

    def to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Converts the result dictionary to a pandas DataFrame."""
        # Filter only array-like items of correct length
        n_samples = len(result["total_signal"])
        data = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray) and len(v) == n_samples:
                data[k] = v
        
        return pd.DataFrame(data)
