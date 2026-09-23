# Reproducibility notes

The cleaned runner has one shared model-preset table and one shared graph/split loader. Each run records effective settings and its exact command. Changing a sparsifier must not silently change the backbone model, training budget, seed, or split.

## Accuracy recipes

`configs/reported_accuracy.yaml` contains imported per-dataset Scaffold-1, Scaffold-K, and Scaffold-Full settings. `--recipe` loads their recorded construction, training, and evaluation flags. See `ACCURACY.md` for coverage and source-table discrepancies. Importing settings is not a new accuracy measurement. Select configurations/checkpoints using validation data, never test accuracy. Preserve run seeds, split fingerprints, actual edge counts, checkpoint policy, refresh cadence, and union size.

Scaffold-1 retains one support throughout training. Scaffold-K refreshes supports and retains a validation-ranked graph bank for union inference. Recipes retain the source checkpoint policy and refresh mode; some historical runs were synchronous. The requested per-support ratio and realized union ratio are different quantities. Sample's number of precomputation forests is independent of the number of inference supports.

Scaffold-Full trains on sparse supports and evaluates on the original full graph. This is the default mode for Fast, Batch, and Sample; default refresh is asynchronous with no periodic deterministic-backbone override. Use `--recipe scaffold-full` for the historical settings. For custom single/multi runs, `--include-full-eval` adds a separate validation-selected full-graph checkpoint, recorded as `view=all`, `model_policy=best-validation` in `multiview.csv`; check `status=OK` before using it. The training target ratio remains sparse. Charge full-graph validation/evaluation when reporting runtime. The `--method full` baseline trains on the full graph too.

## Time measurements

Use fresh result directories for every attempt. Separate data loading, partitioning, feature-weight computation, spanning-forest construction, scoring/selection, Sample weight precomputation, subsequent sampling, training, and evaluation. Report whether caches and JIT kernels were warm. A first Sample support includes precomputation; a successive draw can reuse it. Do not compare that draw time with another method's full cold setup without labeling the distinction.

The launcher records process wall time in `status.json`. Method-native timers retain their original definitions; the launcher does not relabel those timers as complete end-to-end cost. Full-size runtime values from the older environment have not been copied into this cleaned release as newly validated measurements.

The default profiles preserve the source training pipelines, including partition/neighbor training for designated large datasets. Equal epoch counts do not necessarily mean equal optimizer steps or graph coverage across GraphSAGE, GraphSAINT, learned sparsifiers, and full-batch GCN.

## Data and budgets

The shared loader downloads official/public graph assets and verifies the upstream TunedGNN fixed split files where supplied. Cora/CiteSeer/PubMed use TunedGNN class-balanced splits, not automatically the Planetoid public split. Dataset seed and run index are recorded by baseline adapters. All methods use the same loader; raw PyG/OGB download caches may reside in the configured data directory.

For disconnected graphs, a complete spanning forest contains `n-c` edges. Connectivity claims require a feasible edge budget. The source implementations retain legacy budget-fitting behavior below this floor; such runs are not complete spanning supports. Spanning-forest anchors and t-spanners can have unmatched retention and must be labeled accordingly. Learned methods and integer rounding can realize slightly different ratios; report the actual graph size.

## Implementation boundaries

The large-graph AdaGLT/Unified-LTH adapters inherited from the accuracy code are approximations; see `METHODS.md`. They must not be described as the later full-batch native runtime ports. The latter use a different protocol without validation and are not silently mixed into the accuracy runner. The imported accuracy recipes cover Scaffold; baseline defaults remain in `configs/methods/` and the shared preset table.

GPU smoke tests establish that code paths execute. They do not establish statistical accuracy, scalability to every dataset, or equivalence of changed research settings. Use the validation record for precisely what was tested.
