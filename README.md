# ECGTwin

ECGTwin is now structured as an installable `src`-layout Python package with config-driven training, inference, preprocessing, and `pECGMonitor` workflows.

## Install

Create the environment you prefer, then install the package in editable mode:

```sh
pip install -e .
```

For CUDA/conda setups, `environment.yml` remains available as a convenience environment definition.

## Project Layout

- `src/ecgtwin/`: supported application code
- `configs/`: YACS configuration files
- `docs/`: operational documentation
- `research/`: notebooks and ad hoc research scripts
- `data/`: runtime data assets and local datasets

## Common Commands

Train the diffusion model:

```sh
ecgtwin train-diffusion --config configs/experiments/diffusion/dit_ecgtwin.yaml
```

Train the JEPA-style foundation conditioner:

```sh
ecgtwin train-foundation --config configs/experiments/foundation/base.yaml
```

Run inference:

```sh
ecgtwin infer --config configs/experiments/diffusion/dit_ecgtwin.yaml
```

Train the IBE model:

```sh
ecgtwin train-ibe --config configs/experiments/ibe/base.yaml
```

Train the CLIP model:

```sh
ecgtwin train-clip --config configs/experiments/clip/base.yaml
```

Run `pECGMonitor` generation:

```sh
ecgtwin pecg-monitor-generate --config configs/apps/pecg_monitor/base.yaml
```

All commands accept YACS-style overrides after the config path, for example:

```sh
ecgtwin train-diffusion --config configs/experiments/diffusion/dit_ecgtwin.yaml SYSTEM.DEVICE cpu TRAIN.BATCH_SIZE 8
```

## Data Notes

- Dataset preparation guidance lives in `docs/mimic_data.md`.
- Runtime paths, checkpoint locations, and tunable settings are centralized in config files instead of being hardcoded in scripts.

## Documentation

- `docs/README.md`: index of the documentation set
- `docs/architecture.md`: package boundaries and workflow map
- `docs/cli.md`: supported command reference
- `docs/configuration.md`: YACS config tree and override patterns
- `docs/development.md`: maintenance and contribution notes
- `configs/README.md`: config layout guide
- `research/README.md`: what is considered exploratory vs supported
