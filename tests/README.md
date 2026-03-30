# Test Suite

The test suite is currently small and intentionally focused on high-signal regression checks for the refactored repo structure.

## Covered Areas

- YAML config files parse successfully
- core patient/text data semantics remain stable

## Adding Tests

Prefer tests that validate:

- configuration loading and override behavior
- shape and type contracts for data transforms and collate functions
- registry and CLI wiring
- behavior that would be expensive to debug after a structural refactor

