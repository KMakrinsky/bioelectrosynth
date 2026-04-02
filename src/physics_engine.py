import numpy as np
import scipy.signal as signal
import neurokit2 as nk
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict

class SignalSource(ABC):
    """Abstract base class for all signal generators."""
    
    @abstractmethod
    def generate(self, duration: float, fs: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate signal.
        """
        pass

class CorrosionEngine(SignalSource):
    """
    Simulates magnesium implant corrosion including equivalent circuit modeling.
    
    Physics-Aware Mode:
    If enabled, I_corr, Bubble Rate/Amp, and Decay are calculated from R_ct and C_dl.

    Composable circuit:
    If *electrode_circuit* is supplied (an ImpedanceElement tree from circuit_model),
    it is used in impedance calculations instead of the hard-coded Randles formula.
    R_ct_value / C_dl_effective override R_ct / C_dl for Stern-Geary / bubble decay.
    """
    
    def __init__(self, 
                 trend_type: str = 'polynomial',
                 trend_slope: float = 0.0,
                 noise_alpha: float = 1.0,
                 noise_power: float = 1.0,
                 bubble_rate: float = 5.0,
                 bubble_amp_mean: float = 10.0,
                 bubble_decay: float = 0.1,
                 dc_current: float = 0.0,
                 # Equivalent Circuit Parameters
                 use_circuit_model: bool = False,
                 R_s: float = 100.0,      # Ohms
                 R_ct: float = 1000.0,    # Ohms
                 C_dl: float = 10.0e-6,   # Farads (10 uF)
                 R_shunt: float = 10.0,   # Shunt Resistance (Ohms)
                 use_symmetric_electrodes: bool = False,
                 # PHYSICS LINK
                 physics_aware_mode: bool = False,
                 # Composable circuit (new)
                 electrode_circuit=None,   # ImpedanceElement or None
                 R_ct_value: float = None, # Absolute R_ct extracted from circuit
                 C_dl_effective: float = None): # Effective C_dl from CPE (Brug)
        self.trend_type = trend_type
        self.trend_slope = trend_slope
        self.noise_alpha = noise_alpha
        
        # User values (used if physics_aware_mode is False)
        self._user_noise_power = noise_power
        self._user_bubble_rate = bubble_rate
        self._user_bubble_amp_mean = bubble_amp_mean
        self._user_bubble_decay = bubble_decay
        self._user_dc_current = dc_current
        
        # Current active values
        self.noise_power = noise_power
        self.bubble_rate = bubble_rate
        self.bubble_amp_mean = bubble_amp_mean
        self.bubble_decay = bubble_decay
        self.dc_current = dc_current
        
        self.use_circuit_model = use_circuit_model
        self.R_s = R_s
        self.R_ct = R_ct
        self.C_dl = C_dl
        self.R_shunt = R_shunt
        self.use_symmetric_electrodes = use_symmetric_electrodes
        self.physics_aware_mode = physics_aware_mode

        # Composable circuit support
        self.electrode_circuit = electrode_circuit
        if R_ct_value is not None:
            self.R_ct = R_ct_value       # override for Stern-Geary
        if C_dl_effective is not None:
            self.C_dl = C_dl_effective   # override for bubble decay
        
        if self.physics_aware_mode:
            self.recalculate_physics_params()

    def recalculate_physics_params(self):
        """
        Derive corrosion parameters from Impedance (R_ct, C_dl).
        Based on Stern-Geary and Faraday's Laws.
        """
        # 1. Stern-Geary Equation [Stern & Geary, 1957]
        # I_corr = B / R_ct
        # Links electrochemical impedance (R_ct) to corrosion rate (I_corr).
        # Context: [Article on Chitosan/Mg Corrosion & Noise In Vivo]
        # B approx 26mV for Mg/Mg2+ in biological media (Tafel slopes)
        B_tafel = 0.026 # Volts
        I_corr = B_tafel / self.R_ct # Amps
        
        self.dc_current = I_corr
        
        # 2. Bubble Rate (Faraday's Law)
        # H2 evolution consumes 2e- per molecule.
        # Bubble volume at detachment ~ 20 micron radius.
        # V = 4/3 pi r^3.
        # Q_bubble = (V * P / (R*T)) * z * F
        
        # Constants
        r_bub = 20e-6 # 20 microns
        V_bub = (4/3) * np.pi * (r_bub**3)
        P = 101325 # Pa
        R = 8.314
        T = 310 # 37C
        z = 2
        F = 96485
        
        moles = (P * V_bub) / (R * T)
        Q_bub = moles * z * F # Coulombs per bubble
        
        # Bubble nucleation rate from Faradaic charge balance
        self.bubble_rate = I_corr / Q_bub
        
        # Bubble amplitude — small relative to pitting drift (Cottis 2001)
        self.bubble_amp_mean = I_corr * 0.01
        
        # Decay: max of electrical RC and hydrodynamic detachment time
        self.bubble_decay = max(self.R_ct * self.C_dl, 0.08)
        
        # 1/f^2 (Brownian) spectrum dominates metastable pitting noise
        self.noise_alpha = 2.0
        
        # sigma_I ~ 1-10% of I_corr (Cottis 2001, Xin 2008)
        self.noise_power = I_corr * 0.10

    def _generate_colored_noise(self, n_samples: int, alpha: float) -> np.ndarray:
        def generate_channel():
            white = np.random.standard_normal(n_samples)
            if alpha == 0:
                return white
            fft_white = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n_samples)
            with np.errstate(divide='ignore'):
                scale = 1.0 / np.power(np.abs(freqs), alpha / 2.0)
            scale[0] = 0.0
            fft_colored = fft_white * scale
            colored = np.fft.irfft(fft_colored, n=n_samples)
            if np.std(colored) > 0:
                colored = colored / np.std(colored)
            return colored

        c1 = generate_channel()
        if self.use_symmetric_electrodes:
            c2 = generate_channel()
            return c1 - c2
        else:
            return c1

    def _generate_bubbles(self, duration: float, fs: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        def generate_single_channel_bubbles():
            n_bubbles = int(duration * self.bubble_rate)
            if n_bubbles == 0:
                return np.zeros(n_samples), np.zeros(n_samples)
            arrival_indices = np.sort(np.random.choice(n_samples, n_bubbles, replace=False))
            amplitudes = np.random.exponential(self.bubble_amp_mean, size=n_bubbles)
            
            tau_samples = int(self.bubble_decay * fs)
            if tau_samples < 1: tau_samples = 1
            
            impulses = np.zeros(n_samples)
            np.add.at(impulses, arrival_indices, amplitudes)
            mask = np.zeros(n_samples)
            
            vis_threshold = 2.5 * self.noise_power if self.noise_power > 1e-12 else 0.0
            visible_mask = amplitudes > vis_threshold
            visible_indices = arrival_indices[visible_mask]
            mask[visible_indices] = 1.0
            
            decay_factor = np.exp(-1.0 / (fs * self.bubble_decay)) if self.bubble_decay > 0 else 0
            b = [1.0]
            a = [1.0, -decay_factor]
            sig = signal.lfilter(b, a, impulses)
            return sig, mask

        sig1, mask1 = generate_single_channel_bubbles()
        if self.use_symmetric_electrodes:
            sig2, mask2 = generate_single_channel_bubbles()
            total_sig = sig1 - sig2
            total_mask = mask1 + mask2 
            return total_sig, total_mask
        else:
            return sig1, mask1

    def _apply_impedance(self, current_signal: np.ndarray, fs: int) -> np.ndarray:
        n = len(current_signal)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        omega = 2 * np.pi * freqs

        # Electrode impedance: composable circuit or legacy Randles
        if self.electrode_circuit is not None:
            Z_electrode = self.electrode_circuit.impedance(omega)
        else:
            # Legacy single-RC Randles
            j_omega_rc = 1.0 + 1j * omega * self.R_ct * self.C_dl
            Z_electrode = self.R_ct / j_omega_rc
        
        if self.use_symmetric_electrodes:
            Z_total = self.R_s + 2 * Z_electrode + self.R_shunt
        else:
            Z_total = self.R_s + Z_electrode + self.R_shunt
        
        I_fft = np.fft.rfft(current_signal)
        V_fft = I_fft * Z_total
        voltage_signal = np.fft.irfft(V_fft, n=n)
        return voltage_signal

    def _generate_thermal_noise(self, n_samples: int, fs: int) -> np.ndarray:
        k_B = 1.380649e-23
        T = 310.0 
        freqs = np.fft.rfftfreq(n_samples, d=1/fs)
        omega = 2 * np.pi * freqs

        # Re[Z_loop] — composable circuit or legacy Randles
        if self.electrode_circuit is not None:
            Z_electrode = self.electrode_circuit.impedance(omega)
            if self.use_symmetric_electrodes:
                Re_Z = self.R_s + 2 * np.real(Z_electrode) + self.R_shunt
            else:
                Re_Z = self.R_s + np.real(Z_electrode) + self.R_shunt
        else:
            denom = 1.0 + (omega * self.R_ct * self.C_dl)**2
            if self.use_symmetric_electrodes:
                Re_Z = self.R_s + 2 * (self.R_ct / denom) + self.R_shunt
            else:
                Re_Z = self.R_s + (self.R_ct / denom) + self.R_shunt
        
        # Clamp negative Re_Z (can happen with inductive elements at certain frequencies)
        Re_Z = np.maximum(Re_Z, 0.0)

        white_noise = np.random.standard_normal(n_samples)
        fft_white = np.fft.rfft(white_noise)

        # Johnson-Nyquist thermal voltage noise PSD: S_V(f) = 4·k_B·T·Re[Z(f)] [V²/Hz]
        # To shape unit-variance white noise in freq domain:
        #   H(f) = sqrt(S_V(f) · fs)        (spectral shaping filter)
        # Result after IFFT is voltage noise in Volts with correct PSD.
        scale = np.sqrt(4.0 * k_B * T * Re_Z * fs)
        scale[0] = 0  # zero DC
        fft_thermal = fft_white * scale
        thermal_noise = np.fft.irfft(fft_thermal, n=n_samples)
        return thermal_noise  # Volts

    def apply_impedance(self, current_signal: np.ndarray, fs: int) -> np.ndarray:
        return self._apply_impedance(current_signal, fs)
    
    def generate_thermal_noise(self, n_samples: int, fs: int) -> np.ndarray:
        return self._generate_thermal_noise(n_samples, fs)  # Volts

    def generate_current(self, duration: float, fs: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # Auto-recalculate if physics mode enabled
        if self.physics_aware_mode:
            self.recalculate_physics_params()
            
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs

        # --- Trend / DC component ---
        # In symmetric ZRA (two identical electrodes) both electrodes share the
        # same deterministic I_corr and trend → they cancel in the differential
        # measurement.  Only the stochastic part (noise, bubbles) survives.
        if self.use_symmetric_electrodes:
            trend_current = np.zeros(n_samples)
        else:
            trend_current = np.zeros(n_samples) + self.dc_current
            if self.trend_type == 'polynomial':
                drift = (t / duration) ** 2 * (self.dc_current * 0.1)
                trend_current += drift
            elif self.trend_type == 'linear': 
                trend_current += self.trend_slope * t
            elif self.trend_type == 'sine':
                freq = 0.01 
                trend_current += np.sin(2 * np.pi * freq * t) * (self.dc_current * 0.2)

        # --- Stochastic components ---
        # _generate_colored_noise and _generate_bubbles already handle the
        # symmetric case internally (c1 − c2), producing the differential noise
        # between two independent electrode channels.
        colored_noise_current = self._generate_colored_noise(n_samples, self.noise_alpha) * self.noise_power
        bubbles_current, mask_bubbles = self._generate_bubbles(duration, fs, n_samples)
        
        total_current = trend_current + colored_noise_current + bubbles_current
        
        return total_current, {
            "trend_current": trend_current,
            "colored_noise_current": colored_noise_current,
            "bubbles_current": bubbles_current,
            "mask_bubbles": mask_bubbles
        }

    def generate(self, duration: float, fs: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        total_current, comps = self.generate_current(duration, fs)
        n_samples = len(total_current)
        
        if self.use_circuit_model:
            voltage_signal = self._apply_impedance(total_current, fs)
            thermal_noise = self.generate_thermal_noise(n_samples, fs)
            final_signal = voltage_signal + thermal_noise
            comps.update({
                "thermal_noise_voltage": thermal_noise,
                "total_current_source": total_current
            })
            return final_signal, comps
        else:
            return total_current, comps

class BioEngine(SignalSource):
    """
    Simulates biological signals (ECG, EMG) as induced voltages.
    """
    def __init__(self,
                 ecg_enabled: bool = True,
                 ecg_rate: float = 60.0,
                 ecg_amp: float = 0.001, # 1 mV
                 emg_enabled: bool = False,
                 emg_intensity: float = 0.0005, # 0.5 mV
                 emg_burst_prob: float = 0.1, # Probability of burst per second
                 lead_gain: float = 1.0): 
        self.ecg_enabled = ecg_enabled
        self.ecg_rate = ecg_rate
        self.ecg_amp = ecg_amp
        self.emg_enabled = emg_enabled
        self.emg_intensity = emg_intensity
        self.emg_burst_prob = emg_burst_prob
        self.lead_gain = lead_gain

    def generate(self, duration: float, fs: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs
        
        # ECG
        ecg_signal = np.zeros(n_samples)
        if self.ecg_enabled:
            # Use neurokit2 for realistic ECG
            try:
                # Explicitly pass length as int to avoid TypeError with slice indices in some nk versions
                # Also cast sampling_rate to int just in case
                sim_ecg = nk.ecg_simulate(duration=duration, length=n_samples, sampling_rate=int(fs), heart_rate=self.ecg_rate)
                
                # Normalize and scale
                sim_ecg = sim_ecg - np.mean(sim_ecg)
                range_val = np.ptp(sim_ecg)
                if range_val > 0:
                    sim_ecg = sim_ecg / range_val
                
                ecg_signal = sim_ecg * self.ecg_amp * self.lead_gain
            except Exception as e:
                print(f"BioEngine: NeuroKit2 failed ({e}), falling back to sine.")
                # Fallback to simple sine
                ecg_signal = np.sin(2 * np.pi * (self.ecg_rate/60.0) * t) * self.ecg_amp * self.lead_gain

        # EMG (Bursts of high freq noise)
        emg_signal = np.zeros(n_samples)
        mask_emg = np.zeros(n_samples)
        
        if self.emg_enabled:
            # Probability of burst start at each second check
            # Logic: Split into 0.1s chunks, decide if burst
            chunk_size = int(0.1 * fs)
            n_chunks = n_samples // chunk_size
            if n_chunks < 1: n_chunks = 1
            
            for i in range(n_chunks):
                if np.random.random() < (self.emg_burst_prob * 0.1): # Scale prob by time window
                    # Generate burst
                    start_idx = i * chunk_size
                    # Burst length random 0.1 to 0.5s
                    burst_len_sec = np.random.uniform(0.1, 0.5)
                    burst_len = int(burst_len_sec * fs)
                    end_idx = min(start_idx + burst_len, n_samples)
                    
                    if end_idx > start_idx:
                        # Colored noise for EMG (pink/blue mixture, essentially high freq)
                        # Approximated by white noise * envelope
                        burst_noise = np.random.normal(0, 1, end_idx - start_idx)
                        
                        # Apply envelope (attack-sustain-release)
                        env = np.ones_like(burst_noise)
                        attack = int(0.2 * len(env))
                        decay = int(0.2 * len(env))
                        if attack > 0:
                            env[:attack] = np.linspace(0, 1, attack)
                        if decay > 0:
                            env[-decay:] = np.linspace(1, 0, decay)
                            
                        emg_signal[start_idx:end_idx] += burst_noise * env * self.emg_intensity * self.lead_gain
                        mask_emg[start_idx:end_idx] = 1.0

        total_bio = ecg_signal + emg_signal
        
        return total_bio, {
            "ecg_signal": ecg_signal,
            "emg_signal": emg_signal,
            "mask_emg": mask_emg
        }

class SensorEngine(SignalSource):
    """
    Simulates sensor/instrument noise (thermal, mains hum).
    Input params usually in Amps (Current Noise).
    """
    def __init__(self,
                 white_noise_level: float = 1e-9, # 1 nA
                 mains_hum_level: float = 0.0,
                 mains_freq: float = 50.0):
        self.white_noise_level = white_noise_level
        self.mains_hum_level = mains_hum_level
        self.mains_freq = mains_freq

    def generate(self, duration: float, fs: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs
        
        # White Noise (Thermal + Amplifier input noise)
        white_noise = np.random.normal(0, self.white_noise_level, size=n_samples)
        
        # Mains Hum
        hum = np.sin(2 * np.pi * self.mains_freq * t) * self.mains_hum_level
        
        total_sensor = white_noise + hum
        
        return total_sensor, {
            "white_noise_sensor": white_noise,
            "mains_hum_sensor": hum
        }
