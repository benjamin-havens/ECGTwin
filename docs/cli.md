# CLI Reference

The repo exposes a single entrypoint:

```sh
ecgtwin <command> --config <path> [YACS overrides...]
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

