# Architecture

## Top-Level Structure

- `src/ecgtwin/`: supported application code
- `configs/`: versioned workflow configs
- `data/`: runtime assets and local datasets
- `docs/`: operational and maintenance documentation
- `research/`: notebooks and exploratory scripts not treated as supported product surfaces
- `tests/`: lightweight regression checks

## Package Boundaries

### `ecgtwin.cli`

Defines the single public CLI entrypoint. It maps subcommands to package workflows and keeps argument handling thin.

### `ecgtwin.config`

Owns the default YACS tree and config loading. All supported workflows should derive their runtime state from a `CfgNode`, not from hardcoded constants spread across modules.

### `ecgtwin.core`

Contains small cross-cutting runtime helpers. This package intentionally replaces the old catch-all `utils` pattern with narrowly scoped infrastructure helpers.

### `ecgtwin.data`

Owns data semantics:

- patient-feature normalization
- text embedding helpers
- collate functions
- dataset classes
- preprocessing pipelines

The rule is that any logic about how ECG records are represented or transformed belongs here.

### `ecgtwin.models`

Contains neural network definitions and the model factory. Model construction is centralized in `factory.py` so workflow code does not need to know which implementation file a model lives in.

### `ecgtwin.training`

Contains supported training flows:

- diffusion training
- IBE training
- CLIP training

The CLI wrappers in this package are intentionally thin and should mostly be responsible for translating config into runnable objects.

### `ecgtwin.inference`

Contains generation-time orchestration, DDPM sampling, and rendering/export helpers.

### `ecgtwin.apps.pecg_monitor`

Contains the personalized ECG monitoring workflows. This code is treated as a first-class application layer rather than an ad hoc side folder.

## Workflow Map

### Diffusion Training

1. `ecgtwin train-diffusion`
2. Load config
3. Build datasets/dataloader
4. Build model, diffusion scheduler, and IBE model
5. Run training loop in `ecgtwin.training.diffusion`
6. Save logs, checkpoints, and resolved config

### Inference

1. `ecgtwin infer`
2. Load config
3. Load checkpoints and reference sample
4. Build conditioning tensors and text embeddings
5. Sample latent trajectories
6. Decode and render ECG outputs

### Data Preprocessing

Preprocessing workflows live under `ecgtwin.data.preprocess` and are responsible for transforming raw or partially processed datasets into the serialized tensor formats used by training and inference.

## Supported vs Exploratory Code

The code under `src/ecgtwin/` is the supported surface. The material under `research/` is intentionally segregated because it may contain one-off evaluation logic, interactive analysis, or assumptions that are not maintained as part of the primary package API.

