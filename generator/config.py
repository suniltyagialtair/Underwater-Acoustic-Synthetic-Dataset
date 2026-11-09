"""
SKANN-SSL Stage -1 Configuration
================================
Centralized parameters for synthetic waveform generation.
"""

import numpy as np

# =============================================================================
# SIGNAL PARAMETERS
# =============================================================================
FS = 16000
FS_FULL = 32000
DURATION = 5.0
N_SAMPLES = int(FS * DURATION)

FFT_SIZE = N_SAMPLES
FREQ_RES = FS / FFT_SIZE
NYQUIST = FS // 2

MIN_FREQ = 10.0
MAX_FREQ = NYQUIST

# =============================================================================
# SEA NOISE PARAMETERS
# =============================================================================
SEA_STATES = [0, 1, 3, 6]

TURB_START = 10.0
TURB_END = 37.7
LF_END = 200.0
MF_END = 500.0
HF_END = 100000.0

TURB_ANCHOR_1 = (1.0, 108.0)
TURB_ANCHOR_2 = (40.0, 55.0)

SS3_COEFFS = {
    'turb': (-33.10, 108.0),
    'lf': (9.09, 45.28),
    'mf': (-6.57, 83.15),
    'hf': (-17.11, 113.73),
    'f_t': 30.7
}

# =============================================================================
# SHIP NOISE PARAMETERS - CORRECT NON-OVERLAPPING SHAFT RATES
# =============================================================================
SHIP_SNR_DB = 6.0

VESSEL_CLASSES = {
    'small_craft': {
        'f0_range': (15.0, 30.0),       # 900-1800 RPM - NO OVERLAP
        'n_blades': (2, 3),
        'n_harmonics': 8,
        'n_bpf_harmonics': 4,
        'harmonic_decay': 3.0,
        'bpf_decay': 4.0,
        'broadband_level': 0.3,
        'broadband_rolloff': -3.0,
        'cavitation_peak': 5000.0,
    },
    'fishing_vessel': {
        'f0_range': (4.0, 8.0),          # 240-480 RPM - NO OVERLAP
        'n_blades': (3, 4),
        'n_harmonics': 10,
        'n_bpf_harmonics': 5,
        'harmonic_decay': 3.0,
        'bpf_decay': 4.0,
        'broadband_level': 0.4,
        'broadband_rolloff': -4.0,
        'cavitation_peak': 1500.0,
    },
    'cargo_ship': {
        'f0_range': (1.5, 2.5),          # 90-150 RPM
        'n_blades': (4, 5),
        'n_harmonics': 12,
        'n_bpf_harmonics': 6,
        'harmonic_decay': 3.0,
        'bpf_decay': 4.0,
        'broadband_level': 0.5,
        'broadband_rolloff': -5.0,
        'cavitation_peak': 600.0,
    },
    'tanker': {
        'f0_range': (1.0, 1.5),          # 60-90 RPM
        'n_blades': (4, 6),
        'n_harmonics': 15,
        'n_bpf_harmonics': 8,
        'harmonic_decay': 3.0,
        'bpf_decay': 4.0,
        'broadband_level': 0.6,
        'broadband_rolloff': -6.0,
        'cavitation_peak': 400.0,
    }
}

TONAL_PHASE_JITTER = 0.1
TONAL_FREQ_JITTER = 0.02
TONAL_AMP_JITTER = 0.1

GENERATOR_FREQS = [50.0, 100.0, 150.0, 60.0, 120.0, 180.0]
GENERATOR_PROB = 0.7

# =============================================================================
# CAVITATION PARAMETERS
# =============================================================================
CAVITATION_PROB = 0.5
CAVITATION_INTENSITY_RANGE = (0.2, 0.8)
CAVITATION_MOD_FREQ_RANGE = (5.0, 20.0)
CAVITATION_BANDS = [
    {'f_low': 300, 'f_high': 800, 'weight': 0.3},
    {'f_low': 1500, 'f_high': 3000, 'weight': 0.5},
    {'f_low': 4000, 'f_high': 7000, 'weight': 0.2},
]

# =============================================================================
# OTHER PARAMETERS
# =============================================================================
FLOW_NOISE_ROLLOFF = -5.0
FLOW_NOISE_REF_FREQ = 1000.0
N_SYNTHETIC_CLIPS = 500
RANDOM_SEED = 42
DATA_DIR = './data'
WAVEFORM_DIR = f'{DATA_DIR}/waveforms'
TENSOR_DIR = f'{DATA_DIR}/tensors'
METADATA_FILE = f'{DATA_DIR}/metadata.csv'
NO_VESSEL_CLASS_NAME = 'no_vessel'
NO_VESSEL_REPS_PER_SEA_STATE = 120
NO_VESSEL_START_CLIP_ID = 1920
SEA_STATES_DESIGN = [0, 1, 3, 6]
VESSEL_CLASSES_DESIGN = ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']
N_BLADES_OPTIONS = [3, 4, 5]
GEN_FREQ_OPTIONS = [0.0, 50.0]
CAV_INTENSITY_LEVELS = [0.0, 0.3333, 0.6667, 1.0]
FULL_FACTORIAL_REPS = 5
STRUCTURED_DATA_DIR = './structured_dataset'
OLA_OVERLAP = 0.5
OLA_WINDOW = 'hann'
P_REF = 1e-6