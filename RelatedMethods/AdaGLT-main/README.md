# AdaGLT-main

Native small-graph mask learning and retraining. Large-graph learned-score adapters are not the native MLP.

Run from the repository root:

```bash
python run.py --method adaglt --dataset cora --ratio 0.7
python run.py --method adaglt --dataset karate --smoke --workers 2
```

Configuration: [`configs/methods/adaglt.yaml`](../../configs/methods/adaglt.yaml). Shared model/training settings: [`configs/tunedgnn_presets.py`](../../configs/tunedgnn_presets.py). Use `--config deception` for the private scratch-data profile; public defaults use `./data` and all results go under `./results`.

See [`docs/METHODS.md`](../../docs/METHODS.md) for dependency requirements, dataset-specific dispatch, and exact limitations. Model hyperparameters and method-specific search/sampling schedules are separate settings. `--smoke` is not an accuracy recipe.
