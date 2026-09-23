# Settings and run guide

[`configs/reported_accuracy.yaml`](../configs/reported_accuracy.yaml) provides dataset-specific settings for Scaffold-1, Scaffold-K, and Scaffold-Full. Each recipe sets the retention ratio, backbone, scoring parameters, seed, model, training schedule, and evaluation policy.

## Choose an evaluation protocol

| Recipe | Training support | Validation and test |
|---|---|---|
| `scaffold-1` | One fixed sparse support | The same support |
| `scaffold-k` | Refreshed sparse supports | Union of the configured K validation-ranked supports |
| `scaffold-full` | Refreshed sparse supports | Original full graph |

```bash
# Scaffold-Sample is the default method.
python run.py --dataset citeseer --recipe scaffold-1
python run.py --dataset citeseer --recipe scaffold-k
python run.py --dataset citeseer --recipe scaffold-full

# Batch and Fast support the same protocols.
python run.py --dataset citeseer --method scaffold-batch --recipe scaffold-k
python run.py --dataset citeseer --method scaffold-fast --recipe scaffold-full

# Cora recipes: Greedy-1 and Heap-1.
python run.py --dataset cora --method scaffold-greedy --recipe scaffold-1
python run.py --dataset cora --method scaffold-heap --recipe scaffold-1
```

Batch, Fast, and Sample have all three recipes for the other 18 datasets. K is configured per dataset and variant (5 or 10); the number of Sample precomputation forests is a separate setting. The union may contain more edges than an individual support.

Reddit Fast-K and Sample-K produce final `1`, `5`, and `all` views in `multiview.csv`; use the `5` row for their K-support result. Other K recipes evaluate their configured union during training. `--method full` is the comparator that both trains and evaluates on the full graph.

## Default run and custom settings

```bash
# Sample, asynchronous refresh, full-graph evaluation.
python run.py --dataset citeseer --workers auto

# One fixed support, with a custom edge budget.
python run.py --dataset citeseer --mode single --ratio 0.7

# Train on refreshed supports; infer on a union of five supports.
python run.py --dataset citeseer --mode multi --views 5

# Inspect the effective settings without starting training.
python run.py --dataset citeseer --recipe scaffold-k --dry-run
```

Without a recipe, Fast, Batch, and Sample request a new support each epoch and train on the current support until the next is available. `--synchronous` waits for each refresh. Named recipes use their configured refresh interval and synchronous/asynchronous policy; `--no-synchronous` explicitly enables asynchronous refresh.

`--epochs`, `--runs`, `--seed`, `--ratio`, and construction options override recipe values. A recipe fixes the inference protocol, so omit `--mode`, `--views`, and `--include-full-eval` when using `--recipe`. For custom single/multi runs, `--include-full-eval` adds full-graph inference with a separate validation-selected checkpoint.

`--workers auto` uses the available CPU allocation. `--device auto` selects an available visible GPU; use `--device cuda:0` or `--device cpu` to choose explicitly. Sample builds its precomputation artifacts when a matching cache is unavailable.

## Configuration and outputs

The settings file combines `training_defaults`, dataset `training`, `variant_defaults`, and the selected recipe's `parameters`, in that order. Optional recipe `environment` settings include training-time budgets, checked at epoch boundaries.

Data lives in `./data` and outputs in `./results`. Set `SCAFFOLD_DATA_ROOT` to reuse an existing dataset directory, or pass a custom YAML file with `--config`.

Every run saves `resolved.json`, `run.log`, and `status.json`; metric files are written alongside them. `resolved.json` contains the effective training settings and exact command. Multi-view results additionally record the model policy, support count, and realized edge ratio. Use rows with `status=OK`.

The comparison methods use [`configs/tunedgnn_presets.py`](../configs/tunedgnn_presets.py) and their files in [`configs/methods/`](../configs/methods/). See [METHODS.md](METHODS.md) for installation details and method-specific commands.
