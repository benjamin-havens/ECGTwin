# Config Layout

Configuration files are organized by supported workflow rather than by implementation module.

## Structure

- `experiments/diffusion/`: diffusion model training and inference presets
- `experiments/ibe/`: IBE training presets
- `experiments/clip/`: CLIP training presets
- `experiments/privacy/`: privacy-audit presets and base-vector ablation comparisons
- `local/`: machine-specific overlays such as filesystem roots and preferred devices
- `apps/pecg_monitor/`: app-specific presets and prompt/source files

## Conventions

- prefer lowercase file names
- keep YAMLs focused on workflow-specific values
- rely on `ecgtwin.config.defaults` for shared defaults
- use command-line overrides for ad hoc experiments instead of mutating committed YAMLs
- use `MODEL.BASE_VECTOR` for personalization ablations rather than hardcoding behavior in scripts
