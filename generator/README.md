# v5.0.0 generator

The code that produced the released v5.0.0 clips, extracted verbatim from the Colab notebook `SKANN_SSL_V5_Dataset_Generator.ipynb` (28 January 2026):

| File | Notebook cell |
|---|---|
| `config.py` | 6 |
| `SS0CSV.txt`, `SS1CSV.txt`, `SS3CSV.txt`, `SS6CSV.txt` (Knudsen sea-state curves) | 8 |
| `sea_noise.py` | 10 |
| `ship_noise.py` | 12 |
| `generator_colab.py` (the driver that wrote `master_dataset_manifest.csv`) | 14 |

`full_factorial_generator_v5.py` is the same driver as a command-line script. It imports the modules in this folder, and it refuses to run without the four CSV files: without them `sea_noise.py` falls back to default coefficients without warning, and every clip changes.

**Not the release generator:** the 31 KB `ship_noise.py` in SKANN-SSL's `stages/stage_minus1`, which draws a bubble radius and uses three burst regimes. It does not reproduce the dataset.

## Reproduce and check

```
pip install numpy scipy pandas tqdm
cd generator
python full_factorial_generator_v5.py --output ../regenerated
python verify_release.py --dataset <folder with the released waveforms> --out ./_verify
```

The first command regenerates all 12,000 clips and takes hours. `verify_release.py` regenerates 30 clips (every vessel class, sea state and cavitation level, both generator settings, and ambient-only clips) and compares them byte for byte with the released waveforms. On 8 November 2026 all 30 were identical with either driver (Python 3.14.2, NumPy 2.2.6, SciPy 1.16.3, pandas 2.3.3).

In this repository the waveforms are stored with Git LFS: run `git lfs pull` first, or point `--dataset` at the Hugging Face copy.

The code is published as it ran. See the known issues in the main README (cavitation level, cavitation timing and tonal line prominence).

Licence: the code in this folder is MIT ([`LICENSE`](LICENSE)); the dataset and its documentation are CC BY 4.0.
