import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm.auto import tqdm
from sea_noise import SeaNoiseGenerator
from ship_noise import ShipNoiseGenerator

VERSION = "v5.0.0"
FS, N_SAMPLES, P_REF, SNR_DB = 16000, 80000, 1e-6, 6.0
SEA_STATES = [0, 1, 3, 6]
VESSEL_CLASSES = ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']
N_BLADES_OPTIONS, GEN_FREQ_OPTIONS = [3, 4, 5], [0, 50]
CAV_LEVELS = [0.0, 0.3333, 0.6667, 1.0]
EQUIP_POLICY = {0: 0.25, 25: 0.75}

def preprocess(w): x = w - np.mean(w); return (x / (np.sqrt(np.mean(x**2)) + 1e-8)).reshape(1, 1, -1).astype(np.float32)

class ColabDatasetGenerator:
    def __init__(self, output_dir, reps=25, no_vessel_reps=600):
        self.output_dir = Path(output_dir)
        self.waveform_dir, self.tensor_dir = self.output_dir / 'waveforms', self.output_dir / 'tensors'
        self.reps, self.no_vessel_reps = reps, no_vessel_reps
        self.sea_gen, self.ship_gen = SeaNoiseGenerator(FS, N_SAMPLES), ShipNoiseGenerator(FS, N_SAMPLES)
        self.checkpoint_file, self.manifest_file = self.output_dir / 'checkpoint.json', self.output_dir / 'master_dataset_manifest.csv'
    
    def _ensure_dirs(self):
        for d in [self.output_dir, self.waveform_dir, self.tensor_dir]: d.mkdir(parents=True, exist_ok=True)
    
    def _create_tasks(self):
        tasks = []
        total_vessel = len(SEA_STATES) * len(VESSEL_CLASSES) * len(N_BLADES_OPTIONS) * len(GEN_FREQ_OPTIONS) * len(CAV_LEVELS) * self.reps
        rng = np.random.default_rng(42)
        n_zero = int(total_vessel * EQUIP_POLICY[0])
        equip = np.array([0.0] * n_zero + [-1.0] * (total_vessel - n_zero))
        rng.shuffle(equip)
        clip_id = equip_idx = 0
        for ss in SEA_STATES:
            for vc in VESSEL_CLASSES:
                for nb in N_BLADES_OPTIONS:
                    for gf in GEN_FREQ_OPTIONS:
                        for ci in CAV_LEVELS:
                            for rep in range(self.reps):
                                tasks.append({'clip_id': clip_id, 'type': 'vessel', 'sea_state': ss, 'vessel_class': vc,
                                    'n_blades': nb, 'generator_freq': gf, 'cavitation_intensity': ci, 'equipment_policy': equip[equip_idx], 'repeat_index': rep})
                                clip_id += 1; equip_idx += 1
        for ss in SEA_STATES:
            for rep in range(self.no_vessel_reps):
                tasks.append({'clip_id': clip_id, 'type': 'no_vessel', 'sea_state': ss, 'repeat_index': rep})
                clip_id += 1
        return tasks
    
    def _gen_vessel(self, task):
        rng = np.random.default_rng(10000 + task['clip_id'])
        sea = self.sea_gen.generate_frame(sea_state=task['sea_state'], rng=rng)
        sea_rms = np.sqrt(np.mean(sea**2))
        params = self.ship_gen.create_vessel_params(task['vessel_class'], rng)
        params.n_blades, params.blade_pass_freq = task['n_blades'], params.shaft_rate * task['n_blades']
        params.generator_freq = task['generator_freq']
        params.has_cavitation = task['cavitation_intensity'] > 0
        params.cavitation_intensity = task['cavitation_intensity']
        params.equipment_base_freq = task['equipment_policy']
        ship_raw, params = self.ship_gen.generate(params=params, rng=rng)
        ship_rms_raw = np.sqrt(np.mean(ship_raw**2))
        target_rms = sea_rms * (10 ** (SNR_DB / 20))
        scale = target_rms / ship_rms_raw if ship_rms_raw > 0 else 1.0
        ship = ship_raw * scale
        combined = sea + ship
        ship_rms, combined_rms = np.sqrt(np.mean(ship**2)), np.sqrt(np.mean(combined**2))
        wf, tf = f'clip_{task["clip_id"]:06d}.npy', f'tensor_{task["clip_id"]:06d}.npy'
        np.save(self.waveform_dir / wf, combined.astype(np.float32))
        np.save(self.tensor_dir / tf, preprocess(combined))
        return {'clip_id': task['clip_id'], 'repeat_index': task['repeat_index'], 'sea_state': task['sea_state'],
            'vessel_class': task['vessel_class'], 'n_blades': task['n_blades'], 'generator_freq': task['generator_freq'],
            'cavitation_intensity': task['cavitation_intensity'], 'shaft_rate': round(params.shaft_rate, 4),
            'blade_pass_freq': round(params.blade_pass_freq, 4), 'has_cavitation': params.has_cavitation,
            'cavitation_peak_freq': params.cavitation_peak_freq, 'n_cavitation_bursts': params.n_cavitation_bursts,
            'equipment_base_freq': params.equipment_base_freq, 'resonance_freq_1': params.resonance_freq_1,
            'resonance_freq_2': params.resonance_freq_2, 'resonance_freq_3': params.resonance_freq_3,
            'sea_rms_pa': sea_rms, 'ship_rms_pa': ship_rms, 'combined_rms_pa': combined_rms, 'scale_factor': scale,
            'sea_spl_db': round(20*np.log10(sea_rms/P_REF+1e-30), 2), 'ship_spl_db': round(20*np.log10(ship_rms/P_REF+1e-30), 2),
            'combined_spl_db': round(20*np.log10(combined_rms/P_REF+1e-30), 2), 'snr_db': round(20*np.log10(ship_rms/sea_rms+1e-30), 4),
            'filename': wf, 'tensor_path': f'tensors/{tf}', 'waveform_path': f'waveforms/{wf}'}
    
    def _gen_no_vessel(self, task):
        rng = np.random.default_rng(50000 + task['clip_id'])
        sea = self.sea_gen.generate_frame(sea_state=task['sea_state'], rng=rng)
        sea_rms = np.sqrt(np.mean(sea**2))
        wf, tf = f'clip_{task["clip_id"]:06d}.npy', f'tensor_{task["clip_id"]:06d}.npy'
        np.save(self.waveform_dir / wf, sea.astype(np.float32))
        np.save(self.tensor_dir / tf, preprocess(sea))
        return {'clip_id': task['clip_id'], 'repeat_index': task['repeat_index'], 'sea_state': task['sea_state'],
            'vessel_class': 'no_vessel', 'n_blades': 0, 'generator_freq': 0.0, 'cavitation_intensity': 0.0,
            'shaft_rate': 0.0, 'blade_pass_freq': 0.0, 'has_cavitation': False, 'cavitation_peak_freq': 0.0,
            'n_cavitation_bursts': 0, 'equipment_base_freq': 0.0, 'resonance_freq_1': 0.0, 'resonance_freq_2': 0.0,
            'resonance_freq_3': 0.0, 'sea_rms_pa': sea_rms, 'ship_rms_pa': 0.0, 'combined_rms_pa': sea_rms,
            'scale_factor': 0.0, 'sea_spl_db': round(20*np.log10(sea_rms/P_REF+1e-30), 2), 'ship_spl_db': float('-inf'),
            'combined_spl_db': round(20*np.log10(sea_rms/P_REF+1e-30), 2), 'snr_db': float('-inf'),
            'filename': wf, 'tensor_path': f'tensors/{tf}', 'waveform_path': f'waveforms/{wf}'}
    
    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            cp = json.load(open(self.checkpoint_file))
            return cp['last_completed'], cp.get('results', [])
        return -1, []
    
    def _save_checkpoint(self, last, results):
        json.dump({'last_completed': last, 'results': results, 'timestamp': datetime.now().isoformat()}, open(self.checkpoint_file, 'w'))
    
    def generate(self, checkpoint_interval=500):
        self._ensure_dirs()
        tasks = self._create_tasks()
        total = len(tasks)
        vessel_count = sum(1 for t in tasks if t['type'] == 'vessel')
        print("="*60)
        print(f"🚢 SKANN-SSL {VERSION} Dataset Generator")
        print("="*60)
        print(f"📊 Total: {total:,} clips ({vessel_count:,} vessel + {total-vessel_count:,} no-vessel)")
        print(f"📁 Output: {self.output_dir}")
        print()
        last, results = self._load_checkpoint()
        start = last + 1
        if start > 0: print(f"🔄 Resuming from {start:,}/{total:,}\n")
        pbar = tqdm(tasks[start:], initial=start, total=total, desc="🎵 Generating", unit="clip",
            bar_format='{desc}: {percentage:3.1f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        ckpt_count = 0
        for i, task in enumerate(pbar, start=start):
            results.append(self._gen_vessel(task) if task['type'] == 'vessel' else self._gen_no_vessel(task))
            ckpt_count += 1
            pbar.set_postfix_str(f"{task.get('vessel_class', 'no_vessel')[:10]}")
            if ckpt_count >= checkpoint_interval:
                self._save_checkpoint(i, results)
                pbar.write(f"💾 Checkpoint @ {i+1:,}")
                ckpt_count = 0
        pbar.close()
        print("\n📝 Saving manifest...")
        df = pd.DataFrame(results).sort_values('clip_id').reset_index(drop=True)
        df.to_csv(self.manifest_file, index=False)
        if self.checkpoint_file.exists(): self.checkpoint_file.unlink()
        json.dump({'dataset_version': VERSION, 'total_clips': len(df), 'timestamp': datetime.now().isoformat()},
            open(self.output_dir / 'dataset_fingerprint.json', 'w'), indent=2)
        print(f"\n✅ Complete! {len(df):,} clips saved to {self.manifest_file}")
        return df