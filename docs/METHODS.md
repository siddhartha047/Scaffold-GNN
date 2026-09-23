# Methods and settings

Run commands from the repository root after installing the environment. Every method reads the dataset's model/training preset from `configs/tunedgnn_presets.py`; per-method YAML files add method-specific settings. Scaffold's optional `--recipe` overlays the dataset-specific settings in `configs/reported_accuracy.yaml`; see [ACCURACY.md](ACCURACY.md). `--dry-run` prints the resolved configuration. `--smoke` reduces training/search stages and is not an accuracy result.

## Comparison methods

Use `python run.py --method NAME --dataset cora --ratio 0.7` with a name below. Replace the dataset with any of the 19 names in `configs/datasets.yaml`. Full Graph, No Graph, and the two sampling baselines have no retained-edge ratio; their commands record ratio 1.

| Name | Operation / implementation |
|---|---|
| `full` | Full Graph GCN, shared TunedGNN backbone |
| `no-graph` | MLP on node features |
| `spanning-forest` | Fast maximum-weight spanning-forest reference; achieved retention may differ |
| `random` | Uniform retained-edge selection |
| `local-degree` | Local degree-based edge retention |
| `rank-degree` | Degree-ranked traversal/retention |
| `forest-fire` | Randomized graph exploration |
| `scan` | Structural-similarity scoring |
| `l-spar` | Local similarity/degree-constrained retention |
| `g-spar` | Global similarity ranking |
| `l-sim` | Local similarity selection |
| `spectral` | Exact-budget sampling from approximate effective-resistance weights; precompute first |
| `t-spanner` | Las Vegas spanner; graph-dependent achieved retention |
| `dspar` | Degree-based weighted edge sampling with an explicit target budget |
| `mog` | Native mixture-of-experts graph learner, with a shared comparison backbone |
| `adaglt` | Native small-graph mask discovery and sparse retraining; larger-graph adapter described below |
| `unified-lth` | Native small-graph mask search/prune/rewind; larger-graph adapter described below |
| `tuned-graphsage` | TunedGNN GraphSAGE, including dataset-specific large-graph loaders |
| `tuned-graphsaint-rw` | TunedGNN GCN with GraphSAINT random-walk sampling |

The five additional method names are `scaffold-greedy`, `scaffold-heap`, `scaffold-batch`, `scaffold-fast`, and `scaffold-sample`. See the root README for their parameters and single-/multiple-support modes.

## Native methods and larger-graph adapters

The supplied wrappers preserve the source accuracy code's dispatch, rather than substituting an unrelated training loop. Architecture settings alone do not imply identical training procedures.

- **DSpar:** explicit target ratios use the PyTorch weighted sampler; no compiled DSpar extension is needed for these commands. The original sampler with replacement remains available through upstream code and requires building `RelatedMethods/DSpar_tmlr-main/src/setup.py`. The medium-graph comparison uses the TunedGNN model. Smoke tests may take a small partition for non-Cora datasets; normal runs do not enable that smoke-only partition.
- **MoG:** small graphs run the native citation learner. Large graphs use the native Arxiv/Proteins learner modules inside a sparse random-node-partition adapter. The original models in `ogbn_arxiv/` and `ogbn_proteins/` are retained as required dependencies. Each of the three experts uses the requested retention. Partitioning and the mixture can change realized global edge usage.
- **AdaGLT:** the small-graph GCN path performs 400 mask-search epochs by default, then the complete requested retraining schedule. `configs/methods/adaglt.yaml` exposes `ADAGLT_MASK_EPOCHS`. The Cora coefficient remains 0.005. Coauthor-Physics, Questions, Roman-empire, and the five large datasets select a sparse learned-score adapter with projected features and node thresholds. This adapter is **not equivalent to the native 2048-wide edge MLP**.
- **Unified-LTH:** the small-graph path performs two learn/prune/rewind rounds, each with 200 mask-search epochs and a requested fixed-ticket training schedule; graph pruning is separate from weight pruning. Coauthor-Physics, Questions, and the five large datasets select the source's deterministic sparse-ticket adapter. That adapter **does not perform native learned-mask search**, so do not present its results as native Unified-LTH. Final paper recipes must explicitly choose the implementation.
- **Tuned GraphSAGE:** reuses the shared per-dataset GraphSAGE preset. It is the TunedGNN implementation, not the unused plain GraphSAGE wrapper.
- **Tuned GraphSAINT-RW:** uses shared GCN hyperparameters. Walk length, sampler steps, and coverage are set in its wrapper and can be overridden using the documented `BASELINE_GRAPHSAINT_*` environment variables. The normalized loss uses the source's `train_mean` convention. Sampled epochs can contain multiple optimizer steps.

Scaffold recipes are selected with `--recipe`. Comparison methods use the shared training presets and the implementation choices above.

## Spectral precomputation

The effective-resistance implementation is included in `RelatedMethods/Spectral/`. Its dependencies are optional for all other methods.

```bash
# Julia 1.11/1.12; install the pinned project dependencies.
julia --project=RelatedMethods/Spectral -e 'using Pkg; Pkg.instantiate()'

# Small/medium graphs: Julia JL projections with approximate-Cholesky solves.
python scripts/precompute_spectral.py --dataset cora --workers 8
python run.py --method spectral --dataset cora --ratio 0.7
```

For the large-graph C++ backends, install a C++17 compiler, CMake, Eigen3, OpenMP, and ARPACK, then build:

```bash
cmake -S RelatedMethods/Spectral/large_graph -B RelatedMethods/Spectral/build/large_graph
cmake --build RelatedMethods/Spectral/build/large_graph --parallel 8
python scripts/precompute_spectral.py \
  --dataset reddit ogbn-products ogbn-proteins pokec --workers 32
```

The default dispatcher uses Julia with `ceil(4 ln n)` Gaussian projections and solve tolerance `1e-2` for smaller graphs; Products/Proteins/Pokec use 64 Rademacher projections with PCG; Reddit uses the TGT eigensolver with 128 eigenpairs. TGT is a different approximation, not a JL sketch. Artifacts and reports are written below `results/`, separately from downloaded data. A custom artifact can be supplied with `--spectral-artifact PATH`.

The Julia `run.py` utility demonstrates the original sampler with replacement, whose unique-edge count may undershoot its draw budget. The top-level `--method spectral` selects the research **exact-budget** consumer. Precomputation time and sampling time are distinct; charge both when reporting first-use cost.

## Additional dependencies and scope

Use matching PyTorch/PyG binary wheels for `torch-sparse` and `torch-scatter`. The original protein training path additionally uses DGL; install a DGL build compatible with your Python/PyTorch environment before running that profile. No unrelated pruning package is required. The repository targets Linux, where the supplied shell wrappers and CPU-affinity controls are tested.

Use `scripts/smoke.py` for short execution checks before launching longer experiments. See [VALIDATION.md](VALIDATION.md) for tested configurations.
