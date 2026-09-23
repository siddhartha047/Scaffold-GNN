# Comparison implementations

Launch comparisons through the root `run.py` so they receive shared data, splits, TunedGNN model settings, output paths, and the selected GPU. Each method has a small YAML configuration in `configs/methods/` and a README in its source folder.

See [`docs/METHODS.md`](../docs/METHODS.md) for all commands, optional dependencies, and the distinction between native small-graph methods and the larger-graph adapters. In particular, the legacy large-graph Unified-LTH ticket adapter is not native learned-mask pruning. The final reported-accuracy recipe is pending validation.

No datasets, experiment logs, model checkpoints, or Git histories are included here. Original third-party notices are retained.
