# Accuracy settings

[`configs/reported_accuracy.yaml`](../configs/reported_accuracy.yaml) contains the settings traced from `1andKshot_main_results.pdf`, its table generator, and the underlying experiment logs. It also includes a separate full-graph-evaluation campaign. These are imported configurations, not new accuracy measurements from this cleaned repository.

## Run a recipe

```bash
# Scaffold-Sample is the default method.
python run.py --dataset citeseer --recipe scaffold-1
python run.py --dataset citeseer --recipe scaffold-k
python run.py --dataset citeseer --recipe scaffold-full

# Batch and Fast provide the same three protocols.
python run.py --dataset citeseer --method scaffold-batch --recipe scaffold-k
python run.py --dataset citeseer --method scaffold-fast --recipe scaffold-full

# Cora exposes only these two reference recipes.
python run.py --dataset cora --method scaffold-greedy --recipe scaffold-1
python run.py --dataset cora --method scaffold-heap --recipe scaffold-1
```

Add `--dry-run` to inspect the complete command without training. Add `--config deception` to use existing datasets through the local configuration. Outputs always go below `results/`.

| Recipe | Training support | Validation and test |
|---|---|---|
| `scaffold-1` | One frozen support | That same support |
| `scaffold-k` | Refreshed supports | Union of validation-ranked supports; recorded K is 5 or 10 |
| `scaffold-full` | Sparse, refreshed supports | Original full graph |

For Reddit Fast-K and Sample-K, the source evaluates the union at the end of training. Those recipes retain the final `1`, `5`, and `all` views in `multiview.csv`; use the `5` row for K-support results. Other K recipes use their recorded union-evaluation procedure during training. A union can contain more edges than the budget for one support.

Recorded time-limited runs retain their per-run budget through `BASELINE_RUN_TIME_BUDGET_SECONDS`. The loop checks it at epoch boundaries; it is not a strict end-to-end timeout. The budget and any early stop are recorded in the configuration and log.

## Defaults and overrides

Without `--recipe`, the runner uses **Scaffold-Sample with asynchronous support refresh and full-graph evaluation**:

```bash
python run.py --dataset citeseer --workers auto
```

Fast and Batch have the same default evaluation/refresh policy. A refresh is requested each epoch; training continues on the current support while the next is built. `--synchronous` waits for construction instead. `--mode single` freezes one support, and `--mode multi --views 5` requests union inference.

Named recipes preserve the historical synchronous/asynchronous choice, refresh interval, objective coefficients, backbone, Sample weighting rule, training schedule, and seed. Thus, the default refresh policy does not silently change a reported recipe. Sample weights are built locally when no matching cache exists.

Explicit `--epochs`, `--runs`, `--seed`, `--ratio`, and construction options override recipe values. For example, `--no-synchronous` switches a synchronous recipe to asynchronous refresh. Such overrides define a new experiment. A recipe fixes its inference protocol, so it cannot be combined with `--mode`, `--views`, or `--include-full-eval`.

Worker counts adapt to the current allocation; the original count is retained as `source_workers`. Use `--workers N` to request it on a suitable allocation. Asynchronous scheduling and different worker counts can affect which supports become available, so imported settings do not promise bitwise reproduction.

The YAML merges `training_defaults`, per-dataset `training`, `variant_defaults`, and the selected recipe's `parameters`, in that order. Each source-backed entry records a relative source identifier and SHA-256 hash. `resolved.json` saves the source settings, explicit construction overrides, final training settings, and exact executed command.

## Coverage and source issues

There are 164 recipes: Batch, Fast, and Sample under all three protocols for 18 datasets, plus Cora Greedy-1 and Heap-1. Cora's two recipes are configured references; the supplied PDF contains no matching Greedy/Heap one-support measurements.

- **150 source-verified recipes:** settings traced to run logs. This status does not mean that the cleaned repository has rerun their accuracy experiments.
- **10 configured full-evaluation recipes:** all three variants for Products, Proteins, and Pokec, plus Reddit Batch. No matching full-evaluation result was found at the current target retention; these entries carry no reported accuracy claim.
- **2 unresolved paper cells:** Fast-K on Roman-empire and Squirrel. The printed means (88.26 and 43.99) match Sample logs. The catalog provides recorded Fast settings with status `paper_cell_unverified`, without assigning those Sample measurements to Fast.
- **2 configured references:** Cora Greedy-1 and Heap-1.

The PDF caption describes K=5, but several selected results use K=10. The recipes retain the actual recorded K. The source table generator also chooses between some older/newer K campaigns by comparing **test means**; this import preserves that provenance and does not relabel it as validation-only hyperparameter selection. Within-run support ranking and any best-checkpoint selection use validation; the Reddit final-view campaign uses the final model state.

The `scaffold-full` block comes from a separate campaign because the supplied main table contains only 1-support and K-support results. The original research tables and logs are not modified by this import. Comparison methods continue to use the shared TunedGNN presets and the implementation-specific settings described in [METHODS.md](METHODS.md).
