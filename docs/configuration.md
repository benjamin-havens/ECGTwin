# Configuration Guide

ECGTwin uses a YACS configuration tree. Each command loads defaults from `ecgtwin.config.defaults`, merges a YAML file, then merges any command-line overrides.

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
- checkpoint root
- output root
- MIMIC raw-data paths
- patient table path
- exclusion list path
- default reference sample

### `DATA`

Dataset inputs and loader behavior:

- primary dataset path
- train/validation/test dataset paths
- resample length
- split/fold settings
- shuffle behavior

### `MODEL`

Architecture and model-family settings:

- selected model name
- experiment name
- whether VAE latents are used
- whether mixed-text conditioning is enabled
- subtrees for DiT, UNet, IBE, and CLIP-specific hyperparameters

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

## Config File Layout

- `configs/experiments/diffusion/`
- `configs/experiments/ibe/`
- `configs/experiments/clip/`
- `configs/apps/pecg_monitor/`

## Override Syntax

YACS overrides are passed as alternating key/value pairs after `--config`.

Example:

```sh
ecgtwin train-ibe --config configs/experiments/ibe/base.yaml SYSTEM.DEVICE cpu TRAIN.BATCH_SIZE 64
```

## Maintenance Rules

- add new tunable behavior to the config tree before wiring it into code
- keep workflow-specific YAMLs small by inheriting reasonable defaults from code
- prefer adding a new subtree to an existing domain node over inventing a one-off top-level key

