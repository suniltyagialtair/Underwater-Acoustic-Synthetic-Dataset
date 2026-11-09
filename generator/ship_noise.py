import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass
from typing import Optional, Tuple
from config import FS, N_SAMPLES, NYQUIST, FREQ_RES, VESSEL_CLASSES, CAVITATION_PROB, CAVITATION_INTENSITY_RANGE

FS_BURST = 200000

@dataclass
class VesselParams:
    vessel_class: str = 'cargo_ship'
    shaft_rate: float = 2.0
    n_blades: int = 4
    blade_pass_freq: float = 8.0
    n_harmonics: int = 12
    n_bpf_harmonics: int = 6
    harmonic_decay: float = 3.0
    bpf_decay: float = 4.0
    broadband_level: float = 0.5
    broadband_rolloff: float = -5.0
    has_cavitation: bool = False
    cavitation_intensity: float = 0.0
    cavitation_peak: float = 600.0
    generator_freq: float = 0.0
    equipment_base_freq: float = 0.0
    resonance_freq_1: float = 0.0
    resonance_freq_2: float = 0.0
    resonance_freq_3: float = 0.0
    cavitation_peak_freq: float = 0.0
    n_cavitation_bursts: int = 0

class ShipNoiseGenerator:
    def __init__(self, fs: int = FS, n_samples: int = N_SAMPLES):
        self.fs, self.n_samples = fs, n_samples
        self.duration = n_samples / fs
        self.nyquist, self.freq_res = fs // 2, fs / n_samples
        self.n_pos_freqs = n_samples // 2 + 1
        self.freq_grid = np.fft.rfftfreq(n_samples, 1/fs)
    
    def create_vessel_params(self, vessel_class: str, rng: Optional[np.random.Generator] = None) -> VesselParams:
        rng = rng or np.random.default_rng()
        vc = VESSEL_CLASSES[vessel_class]
        f0_min, f0_max = vc['f0_range']
        shaft_rate = rng.uniform(f0_min, f0_max)
        n_blades = rng.integers(vc['n_blades'][0], vc['n_blades'][1] + 1)
        has_cav = rng.random() < CAVITATION_PROB
        cav_int = rng.uniform(*CAVITATION_INTENSITY_RANGE) if has_cav else 0.0
        gen_freq = rng.choice([50.0, 60.0]) if rng.random() < 0.7 else 0.0
        return VesselParams(vessel_class=vessel_class, shaft_rate=shaft_rate, n_blades=n_blades,
            blade_pass_freq=shaft_rate * n_blades, n_harmonics=vc['n_harmonics'], n_bpf_harmonics=vc['n_bpf_harmonics'],
            harmonic_decay=vc['harmonic_decay'], bpf_decay=vc['bpf_decay'], broadband_level=vc['broadband_level'],
            broadband_rolloff=vc['broadband_rolloff'], has_cavitation=has_cav, cavitation_intensity=cav_int,
            cavitation_peak=vc['cavitation_peak'], generator_freq=gen_freq)
    
    def _add_tonal(self, spectrum, freq, amp, rng):
        if freq <= 0 or freq >= self.nyquist: return
        freq += rng.uniform(-0.02, 0.02) * freq
        idx = int(round(freq / self.freq_res))
        if 0 < idx < len(spectrum):
            spectrum[idx] += amp * rng.uniform(0.9, 1.1) * np.exp(1j * rng.uniform(0, 2*np.pi))
    
    def _gen_tonal_spectrum(self, params, rng):
        spectrum = np.zeros(self.n_pos_freqs, dtype=complex)
        for h in range(1, params.n_harmonics + 1):
            self._add_tonal(spectrum, h * params.shaft_rate, 10 ** (-params.harmonic_decay * (h-1) / 20), rng)
        for h in range(1, params.n_bpf_harmonics + 1):
            self._add_tonal(spectrum, h * params.blade_pass_freq, 1.5 * 10 ** (-params.bpf_decay * (h-1) / 20), rng)
        if params.generator_freq > 0:
            for h in range(1, 4):
                self._add_tonal(spectrum, h * params.generator_freq, 0.3 * 10 ** (-3.0 * (h-1) / 20), rng)
        return spectrum
    
    def _gen_broadband(self, params, rng):
        freqs = self.freq_grid.copy()
        freqs[0] = 1.0
        amp = params.broadband_level * (freqs / 1000.0) ** (params.broadband_rolloff / 20)
        amp[0] = 0.0
        phases = rng.uniform(0, 2*np.pi, self.n_pos_freqs)
        phases[0] = 0.0
        return amp * np.exp(1j * phases)
    
    def _add_resonances(self, spectrum, params, rng):
        bands = [(50, 150), (100, 300), (200, 500)]  # V5: Always 3 resonances
        resonances = [rng.uniform(lo, hi) for lo, hi in bands]
        params.resonance_freq_1, params.resonance_freq_2, params.resonance_freq_3 = resonances
        for freq in resonances:
            sigma = freq / rng.uniform(10, 30)
            peak = rng.uniform(0.3, 0.8) * np.exp(-0.5 * ((self.freq_grid - freq) / sigma) ** 2)
            spectrum += peak * np.exp(1j * rng.uniform(0, 2*np.pi))
    
    def _add_equipment(self, spectrum, params, rng):
        if params.equipment_base_freq == 0.0: return
        if params.equipment_base_freq == -1.0:
            params.equipment_base_freq = 30.0 if params.generator_freq == 60.0 else 25.0
        for h in range(1, rng.integers(4, 8) + 1):
            self._add_tonal(spectrum, h * params.equipment_base_freq, 0.2 * 10 ** (-3.0 * (h-1) / 20), rng)
    
    def _gen_cavitation(self, params, rng):
        if not params.has_cavitation or params.cavitation_intensity <= 0:
            params.n_cavitation_bursts = params.cavitation_peak_freq = 0
            return np.zeros(self.n_samples)
        n_burst = int(self.duration * FS_BURST)
        burst_sig = np.zeros(n_burst)
        period = 1.0 / params.blade_pass_freq
        n_bursts = int(self.duration / period)
        swell_freq = rng.uniform(0.05, 0.15)  # V5 FIX: Was 0.5 Hz
        swell_phase = rng.uniform(0, 2*np.pi)
        burst_times = []
        for i in range(n_bursts):
            t_nom = i * period
            t_burst = t_nom + 0.1 * np.sin(2*np.pi*swell_freq*t_nom + swell_phase) * period
            if 0 < t_burst < self.duration: burst_times.append(t_burst)
        params.n_cavitation_bursts = len(burst_times)
        params.cavitation_peak_freq = params.cavitation_peak
        for t_burst in burst_times:
            idx = int(t_burst * FS_BURST)
            tau = rng.uniform(50e-6, 200e-6)
            t_local = np.arange(int(tau * FS_BURST * 10)) / FS_BURST
            burst = np.exp(-t_local / tau) * np.sin(2*np.pi * params.cavitation_peak * rng.uniform(0.8, 1.2) * t_local)
            burst *= params.cavitation_intensity * rng.uniform(0.5, 1.5)
            end_idx = min(idx + len(burst), n_burst)
            if end_idx > idx >= 0: burst_sig[idx:end_idx] += burst[:end_idx - idx]
        cav = scipy_signal.decimate(burst_sig, FS_BURST // self.fs, zero_phase=True)
        if len(cav) > self.n_samples: cav = cav[:self.n_samples]
        elif len(cav) < self.n_samples: cav = np.pad(cav, (0, self.n_samples - len(cav)))
        return cav
    
    def generate(self, vessel_class: Optional[str] = None, params: Optional[VesselParams] = None,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, VesselParams]:
        rng = rng or np.random.default_rng()
        if params is None: params = self.create_vessel_params(vessel_class or 'cargo_ship', rng)
        spectrum = self._gen_tonal_spectrum(params, rng) + self._gen_broadband(params, rng)
        self._add_resonances(spectrum, params, rng)
        self._add_equipment(spectrum, params, rng)
        spectral = np.fft.irfft(spectrum, n=self.n_samples) * self.n_samples / np.sqrt(2)
        return spectral + self._gen_cavitation(params, rng), params