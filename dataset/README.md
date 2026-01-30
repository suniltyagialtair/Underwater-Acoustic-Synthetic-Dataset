# Dataset Files

The dataset files are hosted on Google Drive due to their size (~7 GB).

## Download

**[📥 Download SKANN-SSL V5 Dataset](https://drive.google.com/drive/folders/1E6vhPnkY8x8YzZ3a-k6PnL_G9gnq5gBo)**

## Contents

After downloading, place the files in this directory:

```
dataset/
├── waveforms/
│   ├── clip_000000.npy
│   ├── clip_000001.npy
│   └── ... (12,000 files)
├── tensors/
│   ├── tensor_000000.npy
│   ├── tensor_000001.npy
│   └── ... (12,000 files)
├── master_dataset_manifest.csv
└── pairing_manifest.csv
```

## File Sizes

| Component | Files | Approximate Size |
|-----------|-------|------------------|
| Waveforms | 12,000 | ~3.6 GB |
| Tensors | 12,000 | ~3.6 GB |
| Manifests | 2 | ~5 MB |
| **Total** | | **~7.2 GB** |

## Verification

After downloading, verify the dataset:

```python
import pandas as pd
from pathlib import Path

data_dir = Path('.')
manifest = pd.read_csv(data_dir / 'master_dataset_manifest.csv')

print(f"Total clips: {len(manifest):,}")
print(f"Vessel clips: {len(manifest[manifest.vessel_class != 'no_vessel']):,}")
print(f"No-vessel clips: {len(manifest[manifest.vessel_class == 'no_vessel']):,}")

# Check files exist
waveforms = list((data_dir / 'waveforms').glob('*.npy'))
tensors = list((data_dir / 'tensors').glob('*.npy'))
print(f"Waveform files: {len(waveforms):,}")
print(f"Tensor files: {len(tensors):,}")
```

Expected output:
```
Total clips: 12,000
Vessel clips: 9,600
No-vessel clips: 2,400
Waveform files: 12,000
Tensor files: 12,000
```
