"""
SKANN-SSL V5 Full Factorial Dataset Generator
==============================================
Extracted from SKANN_SSL_V5_Dataset_Generator.ipynb

This is the production V5 generator with:
- Non-overlapping shaft rate ranges
- Fixed swell frequency (0.05-0.15 Hz)
- Always 3 resonances per clip
- Full factorial design: 4 sea states × 4 vessel classes × 3 blade counts × 2 gen freqs × 4 cav levels × 25 reps = 9,600 vessel clips
- Plus 2,400 no-vessel clips (4 sea states × 600 reps)
- Total: 12,000 clips

Usage:
    generator = ColabDatasetGenerator(
        output_dir='./output',
        reps=25,           # 9,600 vessel clips
        no_vessel_reps=600 # 2,400 no-vessel clips
    )
    df = generator.generate(checkpoint_interval=500)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

# Import the v5.0.0 release modules that sit next to this file: config.py,
# sea_noise.py and ship_noise.py, extracted verbatim from
# SKANN_SSL_V5_Dataset_Generator.ipynb (cells 6, 10 and 12). With them this
# script reproduces the released clips byte for byte (see verify_release.py).
# The 31 KB ship_noise.py in SKANN-SSL/stages/stage_minus1 (bubble radius
# drawn from a log-normal, three burst regimes) is NOT the v5.0.0 release
# generator; importing it gives a different dataset.
import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from sea_noise import SeaNoiseGenerator
from ship_noise import ShipNoiseGenerator

VERSION = 'v5.0.0'


def _check_knudsen_csvs():
    """sea_noise.py reads SS{0,1,3,6}CSV.txt from /content or the working
    directory and silently falls back to default coefficients if they are
    missing, which changes every clip. Refuse to run without them."""
    missing = [f'SS{ss}CSV.txt' for ss in (0, 1, 3, 6)
               if not (Path('/content') / f'SS{ss}CSV.txt').exists()
               and not (Path.cwd() / f'SS{ss}CSV.txt').exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found in the working directory ({Path.cwd()}). "
            f"Run from {_HERE}, which holds the four Knudsen CSV files.")

# =============================================================================
# CONFIGURATION
# =============================================================================
FS = 16000
N_SAMPLES = 80000  # 5 seconds at 16 kHz
P_REF = 1e-6
SNR_DB = 6.0

# Full factorial design factors
SEA_STATES = [0, 1, 3, 6]
VESSEL_CLASSES = ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']
N_BLADES_OPTIONS = [3, 4, 5]
GEN_FREQ_OPTIONS = [0, 50]  # Hz (0 = no generator)
CAV_LEVELS = [0.0, 0.3333, 0.6667, 1.0]

# Equipment policy: 25% have equipment_base_freq=0, rest get random
EQUIP_POLICY = {0: 0.25, 25: 0.75}


def preprocess(waveform: np.ndarray) -> np.ndarray:
    """Normalize waveform to zero-mean unit-variance tensor [1, 1, N]."""
    x = waveform - np.mean(waveform)
    return (x / (np.sqrt(np.mean(x**2)) + 1e-8)).reshape(1, 1, -1).astype(np.float32)


class ColabDatasetGenerator:
    """
    Full factorial dataset generator for SKANN-SSL V5.
    
    Generates:
    - 9,600 vessel clips (4 sea states × 4 classes × 3 blades × 2 gen × 4 cav × 25 reps)
    - 2,400 no-vessel clips (4 sea states × 600 reps)
    - Total: 12,000 clips
    
    Features:
    - Checkpoint/resume capability
    - Progress bar with tqdm
    - Saves both raw waveforms (.npy) and preprocessed tensors (.npy)
    - Generates comprehensive manifest CSV
    """
    
    def __init__(self, output_dir: str, reps: int = 25, no_vessel_reps: int = 600):
        """
        Initialize generator.
        
        Args:
            output_dir: Directory to save outputs
            reps: Repetitions per factorial combination (default 25 → 9,600 vessel clips)
            no_vessel_reps: No-vessel clips per sea state (default 600 → 2,400 total)
        """
        self.output_dir = Path(output_dir)
        self.waveform_dir = self.output_dir / 'waveforms'
        self.tensor_dir = self.output_dir / 'tensors'
        self.reps = reps
        self.no_vessel_reps = no_vessel_reps
        
        # Initialize generators
        _check_knudsen_csvs()
        self.sea_gen = SeaNoiseGenerator(FS, N_SAMPLES)
        self.ship_gen = ShipNoiseGenerator(FS, N_SAMPLES)
        
        # Checkpoint files
        self.checkpoint_file = self.output_dir / 'checkpoint.json'
        self.manifest_file = self.output_dir / 'master_dataset_manifest.csv'
    
    def _ensure_dirs(self):
        """Create output directories."""
        for d in [self.output_dir, self.waveform_dir, self.tensor_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _create_tasks(self) -> list:
        """
        Create full factorial task list.
        
        Returns:
            List of task dictionaries with all generation parameters
        """
        tasks = []
        
        # Calculate total vessel clips for equipment policy
        total_vessel = (len(SEA_STATES) * len(VESSEL_CLASSES) * 
                       len(N_BLADES_OPTIONS) * len(GEN_FREQ_OPTIONS) * 
                       len(CAV_LEVELS) * self.reps)
        
        # Pre-generate equipment policy assignments
        rng = np.random.default_rng(42)
        n_zero = int(total_vessel * EQUIP_POLICY[0])
        equip = np.array([0.0] * n_zero + [-1.0] * (total_vessel - n_zero))
        rng.shuffle(equip)
        
        clip_id = 0
        equip_idx = 0
        
        # Full factorial over vessel parameters
        for ss in SEA_STATES:
            for vc in VESSEL_CLASSES:
                for nb in N_BLADES_OPTIONS:
                    for gf in GEN_FREQ_OPTIONS:
                        for ci in CAV_LEVELS:
                            for rep in range(self.reps):
                                tasks.append({
                                    'clip_id': clip_id,
                                    'type': 'vessel',
                                    'sea_state': ss,
                                    'vessel_class': vc,
                                    'n_blades': nb,
                                    'generator_freq': gf,
                                    'cavitation_intensity': ci,
                                    'equipment_policy': equip[equip_idx],
                                    'repeat_index': rep
                                })
                                clip_id += 1
                                equip_idx += 1
        
        # No-vessel clips
        for ss in SEA_STATES:
            for rep in range(self.no_vessel_reps):
                tasks.append({
                    'clip_id': clip_id,
                    'type': 'no_vessel',
                    'sea_state': ss,
                    'repeat_index': rep
                })
                clip_id += 1
        
        return tasks
    
    def _gen_vessel(self, task: dict) -> dict:
        """Generate a single vessel clip."""
        rng = np.random.default_rng(10000 + task['clip_id'])
        
        # Generate sea noise
        sea = self.sea_gen.generate_frame(sea_state=task['sea_state'], rng=rng)
        sea_rms = np.sqrt(np.mean(sea**2))
        
        # Create vessel parameters
        params = self.ship_gen.create_vessel_params(task['vessel_class'], rng)
        params.n_blades = task['n_blades']
        params.blade_pass_freq = params.shaft_rate * task['n_blades']
        params.generator_freq = task['generator_freq']
        params.has_cavitation = task['cavitation_intensity'] > 0
        params.cavitation_intensity = task['cavitation_intensity']
        params.equipment_base_freq = task['equipment_policy']
        
        # Generate ship noise
        ship_raw, params = self.ship_gen.generate(params=params, rng=rng)
        ship_rms_raw = np.sqrt(np.mean(ship_raw**2))
        
        # Scale to target SNR
        target_rms = sea_rms * (10 ** (SNR_DB / 20))
        scale = target_rms / ship_rms_raw if ship_rms_raw > 0 else 1.0
        ship = ship_raw * scale
        
        # Combine
        combined = sea + ship
        ship_rms = np.sqrt(np.mean(ship**2))
        combined_rms = np.sqrt(np.mean(combined**2))
        
        # Save files
        wf = f'clip_{task["clip_id"]:06d}.npy'
        tf = f'tensor_{task["clip_id"]:06d}.npy'
        np.save(self.waveform_dir / wf, combined.astype(np.float32))
        np.save(self.tensor_dir / tf, preprocess(combined))
        
        # Return metadata
        return {
            'clip_id': task['clip_id'],
            'repeat_index': task['repeat_index'],
            'sea_state': task['sea_state'],
            'vessel_class': task['vessel_class'],
            'n_blades': task['n_blades'],
            'generator_freq': task['generator_freq'],
            'cavitation_intensity': task['cavitation_intensity'],
            'shaft_rate': round(params.shaft_rate, 4),
            'blade_pass_freq': round(params.blade_pass_freq, 4),
            'has_cavitation': params.has_cavitation,
            'cavitation_peak_freq': params.cavitation_peak_freq,
            'n_cavitation_bursts': params.n_cavitation_bursts,
            'equipment_base_freq': params.equipment_base_freq,
            'resonance_freq_1': params.resonance_freq_1,
            'resonance_freq_2': params.resonance_freq_2,
            'resonance_freq_3': params.resonance_freq_3,
            'sea_rms_pa': sea_rms,
            'ship_rms_pa': ship_rms,
            'combined_rms_pa': combined_rms,
            'scale_factor': scale,
            'sea_spl_db': round(20 * np.log10(sea_rms / P_REF + 1e-30), 2),
            'ship_spl_db': round(20 * np.log10(ship_rms / P_REF + 1e-30), 2),
            'combined_spl_db': round(20 * np.log10(combined_rms / P_REF + 1e-30), 2),
            'snr_db': round(20 * np.log10(ship_rms / sea_rms + 1e-30), 4),
            'filename': wf,
            'tensor_path': f'tensors/{tf}',
            'waveform_path': f'waveforms/{wf}'
        }
    
    def _gen_no_vessel(self, task: dict) -> dict:
        """Generate a single no-vessel (ambient sea noise) clip."""
        rng = np.random.default_rng(50000 + task['clip_id'])
        
        # Generate sea noise only
        sea = self.sea_gen.generate_frame(sea_state=task['sea_state'], rng=rng)
        sea_rms = np.sqrt(np.mean(sea**2))
        
        # Save files
        wf = f'clip_{task["clip_id"]:06d}.npy'
        tf = f'tensor_{task["clip_id"]:06d}.npy'
        np.save(self.waveform_dir / wf, sea.astype(np.float32))
        np.save(self.tensor_dir / tf, preprocess(sea))
        
        # Return metadata
        return {
            'clip_id': task['clip_id'],
            'repeat_index': task['repeat_index'],
            'sea_state': task['sea_state'],
            'vessel_class': 'no_vessel',
            'n_blades': 0,
            'generator_freq': 0.0,
            'cavitation_intensity': 0.0,
            'shaft_rate': 0.0,
            'blade_pass_freq': 0.0,
            'has_cavitation': False,
            'cavitation_peak_freq': 0.0,
            'n_cavitation_bursts': 0,
            'equipment_base_freq': 0.0,
            'resonance_freq_1': 0.0,
            'resonance_freq_2': 0.0,
            'resonance_freq_3': 0.0,
            'sea_rms_pa': sea_rms,
            'ship_rms_pa': 0.0,
            'combined_rms_pa': sea_rms,
            'scale_factor': 0.0,
            'sea_spl_db': round(20 * np.log10(sea_rms / P_REF + 1e-30), 2),
            'ship_spl_db': float('-inf'),
            'combined_spl_db': round(20 * np.log10(sea_rms / P_REF + 1e-30), 2),
            'snr_db': float('-inf'),
            'filename': wf,
            'tensor_path': f'tensors/{tf}',
            'waveform_path': f'waveforms/{wf}'
        }
    
    def _load_checkpoint(self) -> tuple:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            cp = json.load(open(self.checkpoint_file))
            return cp['last_completed'], cp.get('results', [])
        return -1, []
    
    def _save_checkpoint(self, last: int, results: list):
        """Save checkpoint."""
        json.dump({
            'last_completed': last,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, open(self.checkpoint_file, 'w'))
    
    def generate(self, checkpoint_interval: int = 500) -> pd.DataFrame:
        """
        Generate the full dataset.
        
        Args:
            checkpoint_interval: Save checkpoint every N clips
            
        Returns:
            DataFrame with manifest
        """
        self._ensure_dirs()
        tasks = self._create_tasks()
        total = len(tasks)
        vessel_count = sum(1 for t in tasks if t['type'] == 'vessel')
        
        print("=" * 60)
        print(f"🚢 SKANN-SSL {VERSION} Dataset Generator")
        print("=" * 60)
        print(f"📊 Total: {total:,} clips ({vessel_count:,} vessel + {total - vessel_count:,} no-vessel)")
        print(f"📁 Output: {self.output_dir}")
        print()
        
        # Resume from checkpoint if exists
        last, results = self._load_checkpoint()
        start = last + 1
        if start > 0:
            print(f"🔄 Resuming from {start:,}/{total:,}\n")
        
        # Progress bar
        pbar = tqdm(
            tasks[start:],
            initial=start,
            total=total,
            desc="🎵 Generating",
            unit="clip",
            bar_format='{desc}: {percentage:3.1f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        ckpt_count = 0
        for i, task in enumerate(pbar, start=start):
            if task['type'] == 'vessel':
                results.append(self._gen_vessel(task))
            else:
                results.append(self._gen_no_vessel(task))
            
            ckpt_count += 1
            pbar.set_postfix_str(f"{task.get('vessel_class', 'no_vessel')[:10]}")
            
            if ckpt_count >= checkpoint_interval:
                self._save_checkpoint(i, results)
                pbar.write(f"💾 Checkpoint @ {i + 1:,}")
                ckpt_count = 0
        
        pbar.close()
        
        # Save final manifest
        print("\n📝 Saving manifest...")
        df = pd.DataFrame(results).sort_values('clip_id').reset_index(drop=True)
        df.to_csv(self.manifest_file, index=False)
        
        # Cleanup checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        # Save fingerprint
        json.dump({
            'dataset_version': VERSION,
            'total_clips': len(df),
            'timestamp': datetime.now().isoformat()
        }, open(self.output_dir / 'dataset_fingerprint.json', 'w'), indent=2)
        
        print(f"\n✅ Complete! {len(df):,} clips saved to {self.manifest_file}")
        return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SKANN-SSL V5 Dataset Generator')
    parser.add_argument('--output', '-o', type=str, default='./v5_dataset',
                        help='Output directory')
    parser.add_argument('--reps', type=int, default=25,
                        help='Repetitions per factorial combination (default: 25)')
    parser.add_argument('--no-vessel-reps', type=int, default=600,
                        help='No-vessel clips per sea state (default: 600)')
    parser.add_argument('--checkpoint', type=int, default=500,
                        help='Checkpoint interval (default: 500)')
    
    args = parser.parse_args()

    # sea_noise.py looks for the Knudsen CSVs in the working directory
    output_dir = Path(args.output).resolve()
    import os
    os.chdir(_HERE)

    generator = ColabDatasetGenerator(
        output_dir=output_dir,
        reps=args.reps,
        no_vessel_reps=args.no_vessel_reps
    )
    
    df = generator.generate(checkpoint_interval=args.checkpoint)
    print(f"\nDataset shape: {df.shape}")
    print(f"Classes: {df['vessel_class'].value_counts().to_dict()}")
