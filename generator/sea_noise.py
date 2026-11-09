import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import warnings
from config import FS, N_SAMPLES, P_REF

TURB_ANCHOR_1 = (10.0, 108.0)
TURB_ANCHOR_2 = (40.0, 55.0)
LF_END = 200.0
MF_END = 500.0
MIN_FREQ = 10.0

class KnudsenModel:
    def __init__(self, csv_path: Optional[str] = None, sea_state: int = 3):
        self.sea_state = sea_state
        self.coefficients = {}
        self.f_t = None
        if csv_path is not None:
            self._fit_from_csv(csv_path)
        else:
            self._set_default_coefficients()
    
    def _set_default_coefficients(self):
        self.coefficients = {'turb': (-88.01, 108.0), 'lf': (13.30, 36.29), 'mf': (-0.033, 66.96), 'hf': (-16.22, 110.65)}
        self.f_t = 35.1
    
    def _fit_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path, header=None, names=['idx', 'f', 'NL']).sort_values('f').reset_index(drop=True)
        f1, NL1 = TURB_ANCHOR_1
        f2, NL2 = TURB_ANCHOR_2
        a_turb = (NL2 - NL1) / (np.log10(f2) - np.log10(f1))
        b_turb = NL1 - a_turb * np.log10(f1)
        lf_mask = (df['f'] >= 100.0) & (df['f'] <= 200.0)
        if lf_mask.sum() >= 2:
            lf_data = df[lf_mask]
            x_lf = np.log10(lf_data['f'].values)
            A_lf = np.vstack([x_lf, np.ones_like(x_lf)]).T
            a_lf, b_lf = np.linalg.lstsq(A_lf, lf_data['NL'].values, rcond=None)[0]
        else:
            a_lf, b_lf = 13.30, 36.29
        self.f_t = 10 ** ((b_turb - b_lf) / (a_lf - a_turb)) if abs(a_lf - a_turb) > 1e-6 else 35.0
        hf_mask = (df['f'] >= 500.0)
        if hf_mask.sum() >= 2:
            hf_data = df[hf_mask]
            x_hf = np.log10(hf_data['f'].values)
            A_hf = np.vstack([x_hf, np.ones_like(x_hf)]).T
            a_hf, b_hf_raw = np.linalg.lstsq(A_hf, hf_data['NL'].values, rcond=None)[0]
        else:
            a_hf, b_hf_raw = -16.22, 110.65
        NL_200 = a_lf * np.log10(200.0) + b_lf
        NL_500 = a_hf * np.log10(500.0) + b_hf_raw
        a_mf = (NL_500 - NL_200) / (np.log10(500.0) - np.log10(200.0))
        b_mf = NL_200 - a_mf * np.log10(200.0)
        b_hf = NL_500 - a_hf * np.log10(500.0)
        self.coefficients = {'turb': (a_turb, b_turb), 'lf': (a_lf, b_lf), 'mf': (a_mf, b_mf), 'hf': (a_hf, b_hf)}
    
    def nl(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        f = np.atleast_1d(np.asarray(f, dtype=float))
        nl = np.zeros_like(f)
        a_lf, b_lf = self.coefficients['lf']
        a_mf, b_mf = self.coefficients['mf']
        a_hf, b_hf = self.coefficients['hf']
        nl_plateau = a_lf * np.log10(self.f_t) + b_lf
        nl[(f >= MIN_FREQ) & (f < self.f_t)] = nl_plateau
        mask_lf = (f >= self.f_t) & (f < LF_END)
        nl[mask_lf] = a_lf * np.log10(f[mask_lf]) + b_lf
        mask_mf = (f >= LF_END) & (f < MF_END)
        nl[mask_mf] = a_mf * np.log10(f[mask_mf]) + b_mf
        nl[f >= MF_END] = a_hf * np.log10(f[f >= MF_END]) + b_hf
        return nl.squeeze()
    
    def psd(self, f): return 10 ** (self.nl(f) / 10) * 1e-12

class SeaNoiseGenerator:
    def __init__(self, fs: int = FS, n_samples: int = N_SAMPLES):
        self.fs, self.n_samples = fs, n_samples
        self.nyquist, self.freq_res = fs // 2, fs / n_samples
        self.n_pos_freqs = n_samples // 2 + 1
        self.freq_grid = np.fft.rfftfreq(n_samples, 1/fs)
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        for ss in [0, 1, 3, 6]:
            for d in [Path('/content'), Path.cwd()]:
                p = d / f'SS{ss}CSV.txt'
                if p.exists():
                    self.models[ss] = KnudsenModel(str(p), sea_state=ss)
                    break
            else:
                self.models[ss] = KnudsenModel(sea_state=ss)
    
    def generate_frame(self, sea_state: int = 3, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        model = self.models.get(sea_state, self.models[3])
        freqs = self.freq_grid.copy()
        freqs[0] = 1.0
        psd = model.psd(freqs)
        psd[0] = 0.0
        amplitude = np.sqrt(psd * self.freq_res)
        phases = rng.uniform(0, 2 * np.pi, self.n_pos_freqs)
        phases[0] = 0.0
        if self.n_samples % 2 == 0: phases[-1] = 0.0
        spectrum = amplitude * np.exp(1j * phases)
        return np.fft.irfft(spectrum, n=self.n_samples) * self.n_samples / np.sqrt(2)