# CLI Reference

The repo exposes a single entrypoint:

```sh
ecgtwin <command> --config <path> [--config <path> ...] [YACS overrides...]
```

## Training Commands

### Diffusion

```sh
ecgtwin train-diffusion --config configs/experiments/diffusion/dit_ecgtwin.yaml
```

Runs the paired ECG diffusion training flow.

### IBE

```sh
ecgtwin train-ibe --config configs/experiments/ibe/base.yaml
```

Runs the individual base extractor training flow.

### CLIP

```sh
ecgtwin train-clip --config configs/experiments/clip/base.yaml
```

Runs the ECG/text alignment training flow.

## Inference

```sh
ecgtwin infer --config configs/experiments/diffusion/dit_ecgtwin.yaml
```

Generates ECG samples using the configured diffusion checkpoint and reference sample.

## Privacy Audit

### Build Audit Splits

```sh
ecgtwin privacy-splits --config configs/experiments/privacy/base.yaml
```

Builds the member and nonmember audit manifest from serialized ECG datasets.

### Generate Synthetic Pools

```sh
ecgtwin privacy-generate --config configs/experiments/privacy/base.yaml
```

Generates per-record synthetic latent pools for black-box MIA and DOMIAS-style scoring.

### Run Full Audit

```sh
ecgtwin privacy-audit --config configs/experiments/privacy/base.yaml
```

Runs black-box MIA, white-box IBE and diffusion MIA, synthetic-only DOMIAS scoring, and a model-inversion-style reconstruction attack that reuses each record's synthetic pool.
If `PRIVACY.GPU_IDS` is set, the audit parallelizes across one process per listed GPU.

## Preprocessing

### Pair Serialized ECGs

```sh
ecgtwin preprocess-pair --config <config>
```

Builds chronologically ordered paired ECG samples from serialized tensor records.

### Store Text Embeddings

```sh
ecgtwin preprocess-embed --config <config>
```

Adds text embeddings to serialized tensor datasets.

### Encode With VAE

```sh
ecgtwin encode-vae --config <config>
```

Encodes waveform datasets into the latent space expected by downstream training jobs.

## `pECGMonitor`

### Generate Personalized Samples

```sh
ecgtwin pecg-monitor-generate --config configs/apps/pecg_monitor/base.yaml
```

### Train Classifier

```sh
ecgtwin pecg-monitor-train-classifier --config configs/apps/pecg_monitor/base.yaml
```

### Test Classifier

```sh
ecgtwin pecg-monitor-test-classifier --config configs/apps/pecg_monitor/base.yaml
```

## Override Examples

Run on CPU:

```sh
ecgtwin infer --config configs/experiments/diffusion/dit_ecgtwin.yaml SYSTEM.DEVICE cpu
```

Change batch size and output path:

```sh
ecgtwin train-diffusion --config configs/experiments/diffusion/dit_ecgtwin.yaml TRAIN.BATCH_SIZE 8 PATHS.CHECKPOINTS_DIR tmp/checkpoints
```

Enable the noisy base-vector ablation during inference or privacy runs:

```sh
ecgtwin privacy-audit --config configs/experiments/privacy/base.yaml MODEL.BASE_VECTOR.MODE noise MODEL.BASE_VECTOR.NOISE_STD 0.25
```

Run the privacy audit across GPUs 0, 2, and 3 with layered configs:

```sh
ecgtwin privacy-audit \
  --config configs/experiments/privacy/base.yaml \
  --config configs/local/havens3_privacy_full.yaml \
  --config configs/local/havens3_privacy_gpus_0_2_3.yaml
```

Mix a model config and a data config in order:

```sh
ecgtwin train-diffusion --config configs/experiments/diffusion/dit_ecgtwin.yaml --config configs/experiments/privacy/remove_base_vector.yaml --config path/to/data.yaml
```
