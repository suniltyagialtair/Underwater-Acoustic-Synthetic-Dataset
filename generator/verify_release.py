"""
Regenerate released v5.0.0 clips and compare them byte for byte with the
published waveforms.

Run from this folder (sea_noise.py reads the Knudsen CSVs from the working
directory):

    python verify_release.py --dataset .. --out ./_verify
    python verify_release.py --dataset .. --out ./_verify --driver full_factorial_generator_v5
    python verify_release.py --dataset .. --out ./_verify --clips 0,75,1581

--dataset is the folder that holds master_dataset_manifest.csv and the
waveforms it points to (this repository's root, or the Hugging Face copy).
Regenerated clips are written under --out and can be deleted afterwards.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# 30 distinct clips: every vessel class, every sea state, all four cavitation
# levels, both generator settings, and ambient-only clips.
DEFAULT_CLIPS = [0, 62, 75, 99, 549, 1157, 1250, 1270, 2168, 2749, 2883, 3333,
                 4494, 4795, 5555, 5989, 6564, 7449, 7662, 7884, 7991, 8000,
                 8382, 8595, 8764, 9060, 9599, 9700, 10850, 11999]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    ap.add_argument('--dataset', required=True, help='folder with master_dataset_manifest.csv and the waveforms')
    ap.add_argument('--out', required=True, help='scratch folder for regenerated clips')
    ap.add_argument('--driver', default='generator_colab', choices=['generator_colab', 'full_factorial_generator_v5'])
    ap.add_argument('--clips', default=','.join(map(str, DEFAULT_CLIPS)), help='comma-separated clip_ids')
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    gen = importlib.import_module(args.driver).ColabDatasetGenerator(output_dir=out, reps=25, no_vessel_reps=600)
    gen._ensure_dirs()
    tasks = {t['clip_id']: t for t in gen._create_tasks()}
    manifest = pd.read_csv(dataset / 'master_dataset_manifest.csv').set_index('clip_id')

    ids = [int(c) for c in args.clips.split(',')]
    ok = 0
    for cid in ids:
        task = tasks[cid]
        gen._gen_vessel(task) if task['type'] == 'vessel' else gen._gen_no_vessel(task)
        mine = np.load(out / 'waveforms' / f'clip_{cid:06d}.npy')
        path = dataset / manifest.loc[cid, 'waveform_path']
        if path.read_bytes()[:40].startswith(b'version https://git-lfs'):
            sys.exit(f"{path} is a Git LFS pointer, not the waveform. Run 'git lfs pull' "
                     f"or point --dataset at the Hugging Face copy.")
        released = np.load(path)
        same = mine.dtype == released.dtype and mine.shape == released.shape and mine.tobytes() == released.tobytes()
        ok += same
        print(f"clip {cid:5d}  {manifest.loc[cid, 'vessel_class']:14s}  sea state {manifest.loc[cid, 'sea_state']}  "
              f"cavitation {manifest.loc[cid, 'cavitation_intensity']:.4f}  {'identical' if same else 'DIFFERENT'}")
    print(f"\n{ok} of {len(ids)} clips byte-identical to the released waveforms ({args.driver}).")
    sys.exit(0 if ok == len(ids) else 1)


if __name__ == '__main__':
    main()
