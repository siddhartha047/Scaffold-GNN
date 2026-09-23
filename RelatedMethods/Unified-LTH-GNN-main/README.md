# Unified-LTH-GNN-main

Native small-graph pruning/rewinding; the legacy large-graph deterministic-ticket adapter is not native learned-mask pruning.

Run from the repository root:

```bash
python run.py --method unified-lth --dataset cora --ratio 0.7
python run.py --method unified-lth --dataset karate --smoke --workers 2
```

Configuration: [`configs/methods/unified-lth.yaml`](../../configs/methods/unified-lth.yaml). Shared model/training settings: [`configs/tunedgnn_presets.py`](../../configs/tunedgnn_presets.py). Use `--config deception` for the private scratch-data profile; public defaults use `./data` and all results go under `./results`.

See [`docs/METHODS.md`](../../docs/METHODS.md) for dependency requirements, dataset-specific dispatch, and exact limitations. Model hyperparameters and method-specific search/sampling schedules are separate settings. `--smoke` is not an accuracy recipe.
