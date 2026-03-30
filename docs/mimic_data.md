# MIMIC Data Preparation

This document explains how the refactored repo expects MIMIC-based training data to be prepared.

## Expected Outputs

Typical downstream workflows consume serialized tensor datasets such as:

- `Mimic_vae.pt`
- `Mimic_vae_multi_nomic.pt`
- `paired_Mimic_vae_multi_nomic.pt`
- `paired_Mimic_vae_mix_nomic.pt`

The exact filenames are configurable, but these are the historical dataset names used by the existing configs.

## Option 1: Download Prepared Data

If you already have prepared tensors, place them under your local data root and point the relevant config values at them.

The key config entries are usually:

- `DATA.DATASET_PATH`
- `DATA.TRAIN_DATASET_PATH`
- `DATA.TEST_DATASET_PATH`
- `PATHS.DATA_ROOT`

## Option 2: Build Locally

### Step 1: Encode Raw ECGs Into VAE Latents

Prepare access to:

- the raw MIMIC-IV-ECG waveform dataset
- the MIMIC patient table
- the repository exclusion list

Then run:

```sh
ecgtwin encode-vae --config configs/experiments/diffusion/dit_ecgtwin.yaml \
  PATHS.MIMIC_ROOT /path/to/mimic-iv-ecg \
  PATHS.PATIENTS_CSV /path/to/patients.csv \
  PATHS.OUTPUT_DIR data
```

This uses `ecgtwin.data.preprocess.vae_encoding` and writes latent tensors into the configured output directory.

### Step 2: Add Text Embeddings

Once the latent dataset exists, add text embeddings:

```sh
ecgtwin preprocess-embed --config configs/experiments/diffusion/dit_ecgtwin.yaml \
  DATA.DATASET_PATH data/Mimic_vae.pt \
  DATA.TRAIN_DATASET_PATH data/Mimic_vae_multi_nomic.pt
```

If you want whole-report mixed embeddings instead of split diagnosis embeddings, set:

```sh
MODEL.MIX_TEXT true
```

### Step 3: Build Paired Datasets

Build chronologically ordered subject pairs:

```sh
ecgtwin preprocess-pair --config configs/experiments/diffusion/dit_ecgtwin.yaml \
  DATA.DATASET_PATH data/Mimic_vae_multi_nomic.pt \
  DATA.TRAIN_DATASET_PATH data/paired_Mimic_vae_multi_nomic.pt
```

For mixed-text paired datasets used by the adaLN-style experiments:

```sh
ecgtwin preprocess-pair --config configs/experiments/diffusion/dit_adaln.yaml \
  DATA.DATASET_PATH data/Mimic_vae_mix_nomic.pt \
  DATA.TRAIN_DATASET_PATH data/paired_Mimic_vae_mix_nomic.pt
```

## Operational Notes

- The raw-data adapters expect the exclusion list at `data/exclude_list.txt` unless you override `PATHS.EXCLUDE_LIST`.
- Paths in the repo configs are examples; in practice, most users will override them for their local storage layout.
- Preprocessing outputs are regular serialized tensor datasets, so it is normal to keep them outside version control.
