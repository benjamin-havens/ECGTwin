# Development Notes

## Documentation Expectations

This repo uses two documentation layers:

- in-file docstrings and targeted comments for local code comprehension
- higher-level docs and readmes for workflow and architecture guidance

When you change behavior or structure, update both when needed.

## Adding New Functionality

### New Models

1. Add the implementation to `ecgtwin.models`
2. Register it in `ecgtwin.models.factory`
3. Add config keys under `MODEL`
4. Add or update a config YAML
5. Add at least one smoke or semantic test

### New Workflows

1. Add the implementation module under the correct subsystem
2. Add a thin CLI wrapper if it is part of the supported surface
3. Add config support
4. Document the command and config in `docs/`

## Avoid Reintroducing `utils`

The refactor intentionally removed the generic `utils` bucket. If a helper has no clear home, first decide whether it is:

- config-related
- runtime/logging-related
- data-related
- inference-related
- training-related
- app-specific

If it still does not fit, the design likely needs another pass.

## Tests

The current tests are lightweight and focus on:

- config parsing
- core data semantics

Add more tests around any new behavior that would be costly to debug if it regressed.

## Research Assets

Notebooks and one-off scripts live under `research/`. They are not treated as stable public interfaces. If a research script becomes a maintained workflow, migrate it into `src/ecgtwin/` and document it in `docs/cli.md`.

