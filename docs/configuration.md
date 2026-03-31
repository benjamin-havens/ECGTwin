# Configuration Guide

ECGTwin uses a YACS configuration tree. Each command loads defaults from `ecgtwin.config.defaults`, merges one or more YAML files in the order they are provided, then merges any command-line overrides.

## Core Principles

- defaults live in code
- workflow-specific values live in `configs/`
- command-line overrides are for temporary run-specific changes
- path strings and tunable parameters should not be hardcoded in workflow modules

## Major Config Nodes

### `SYSTEM`

Runtime-level execution settings:

- device
- seed
- dataloader worker count
- pin-memory and AMP toggles

### `PATHS`

Shared filesystem locations:

- data root
- serialized tensor dataset root
- checkpoint root
- output root
- MIMIC raw-data paths
- patient table path
- exclusion list path
- default reference sample

### `DATA`

Dataset inputs and loader behavior:

- primary dataset filename or relative path under `PATHS.SERIALIZED_DATA_ROOT`
- train/validation/test dataset filenames or relative paths under `PATHS.SERIALIZED_DATA_ROOT`
- resample length
- split/fold settings
- shuffle behavior

### `MODEL`

Architecture and model-family settings:

- selected model name
- experiment name
- whether VAE latents are used
- whether mixed-text conditioning is enabled
- subtrees for DiT, UNet, IBE, CLIP, and `BASE_VECTOR` personalization controls

### `MODEL.BASE_VECTOR`

Conditioning ablations and personalization controls:

- base-vector mode: `standard`, `remove`, `noise`, or `bottleneck`
- optional Gaussian noise level
- bottleneck width for reduced-dimension personalized conditioning
- feature-mask probability used during diffusion training
- whether the ablation should also be applied during inference and privacy evaluation

### `DIFFUSION`

Scheduler-level settings:

- number of training steps
- beta schedule bounds
- inference timestep count

### `TRAIN`

Training-loop parameters:

- task name
- epochs
- batch size
- mini-batch size
- learning rate
- weight decay
- classifier weighting and load-from-pretrain options

### `CHECKPOINTS`

Known model artifact paths:

- diffusion checkpoint
- VAE checkpoint
- IBE checkpoint
- CLIP checkpoint

### `INFERENCE`

User-facing generation inputs:

- save directory
- generation batch count
- target demographics
- target text prompt

### `APPS.PECG_MONITOR`

App-specific paths and runtime parameters for the personalized monitoring workflows.

### `PRIVACY`

Privacy-audit inputs and attack settings:

- member and nonmember dataset paths
- audit output directory and experiment name
- attack levels: patient and record
- feature space used for black-box scoring
- synthetic pool sizing
- `GPU_IDS` for process-per-GPU privacy-audit parallelism
- black-box, white-box, DOMIAS, and reconstruction attack hyperparameters

## Config File Layout

- `configs/experiments/diffusion/`
- `configs/experiments/ibe/`
- `configs/experiments/clip/`
- `configs/experiments/privacy/`
- `configs/apps/pecg_monitor/`

## Override Syntax

YACS overrides are passed as alternating key/value pairs after the config list.

Example:

```sh
ecgtwin train-ibe --config configs/experiments/ibe/base.yaml SYSTEM.DEVICE cpu TRAIN.BATCH_SIZE 64
```

Serialized `.pt` dataset paths are resolved relative to `PATHS.SERIALIZED_DATA_ROOT` unless you pass an absolute path.

You can also layer partial configs:

```sh
ecgtwin train-diffusion --config configs/experiments/diffusion/dit_ecgtwin.yaml --config path/to/model.yaml --config path/to/data.yaml
```

Later config files win if they set the same keys.

## Maintenance Rules

- add new tunable behavior to the config tree before wiring it into code
- keep workflow-specific YAMLs small by inheriting reasonable defaults from code
- prefer adding a new subtree to an existing domain node over inventing a one-off top-level key
