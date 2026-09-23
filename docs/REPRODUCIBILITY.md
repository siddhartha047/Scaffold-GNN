# Experiment protocols

All methods share the TunedGNN model presets and graph/split loader. Each run saves its effective settings and exact command in `resolved.json`.

## Evaluation

- **Scaffold-1:** one support throughout training, validation, and test.
- **Scaffold-K:** refreshed training supports and inference on a validation-ranked support union. Report both the per-support retention and realized union size.
- **Scaffold-Full:** sparse training with evaluation on the original graph.

Use the [settings guide](ACCURACY.md) to select a recipe. Recipes specify checkpoint selection, refresh cadence, seed, and synchronous/asynchronous construction. For custom runs, `--include-full-eval` adds a separate full-graph result and validation-selected checkpoint. Sample's precomputation forest count is independent of its inference support count.

## Timing

Separate data loading, partitioning, edge-weight computation, backbone construction, edge selection, Sample precomputation, subsequent sampling, training, and evaluation. Record whether caches and JIT kernels were warm. Keep `results/cache` for warm-start measurements; choose a fresh cache directory for cold-start measurements.

Process wall time is saved in `status.json`; component timers are printed in `run.log`. Include full-graph evaluation and precomputation when reporting end-to-end cost. With asynchronous construction, CPU work can overlap GPU training, so summed component time can exceed elapsed wall time.

Large-graph presets can use partition or neighbor training. Equal epoch counts across full-batch, sampled, and learned-sparsifier pipelines need not imply equal optimizer steps.

## Splits and edge budgets

The loader downloads public graph assets and verifies fixed split files where supplied. Cora, CiteSeer, and PubMed use TunedGNN class-balanced splits. Preserve seeds and split fingerprints when comparing methods.

A complete spanning forest on a graph with `n` nodes and `c` connected components needs `n-c` edges. Below that budget, partial-support policies cannot preserve connectivity. Spanning-forest and spanner references can have unmatched retention; report their actual edge counts. Weighted and unweighted path settings are distinct experimental choices.

## Comparison implementations

[METHODS.md](METHODS.md) describes the native methods and larger-graph adapters. In particular, the large-graph AdaGLT and Unified-LTH adapters differ from their native mask-learning algorithms. Keep these implementation choices explicit when reporting results.
