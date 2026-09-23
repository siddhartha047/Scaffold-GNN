# Scaffold

**Sparse graph supports for GNN training and inference.** Scaffold starts from a spanning forest and retains edges that improve supporting-path dilation and congestion.

<p align="center"><img src="docs/images/pipeline.png" width="850" alt="Scaffold-1 reuses one sparse support; Scaffold-K trains on refreshed supports and infers on their union."></p>

This repository contains Scaffold, shared TunedGNN settings, and the comparison methods in `RelatedMethods/`. Dataset-specific settings for Scaffold-1, Scaffold-K, and Scaffold-Full are in [`configs/reported_accuracy.yaml`](configs/reported_accuracy.yaml); see the [settings and run guide](docs/ACCURACY.md).

## Install

Use Python 3.11 or 3.12 on Linux. Install a CUDA-compatible [PyTorch](https://pytorch.org/get-started/locally/) and the matching [PyG extension wheels](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) first. `torch-sparse` and `torch-scatter` are required by the comparison models and partitioning; their wheels must match your PyTorch/CUDA versions.

```bash
python -m pip install -e .
python scripts/check_environment.py
python run.py --list
python run.py --dataset karate --smoke --workers 2
```

The runner selects the visible CUDA device with the most free memory. For a CPU-only test, add `--device cpu`. It never requests a Slurm allocation. `--workers auto` respects CPU affinity and `SLURM_CPUS_PER_TASK`; explicit worker counts are capped at the available allocation. Use the same Python environment for the launcher and its subprocesses.

For five-epoch Cora checks of every method, use `python scripts/smoke.py --dataset cora --epochs 5 --all-methods`. Spectral also needs Julia and its dependencies, described in [METHODS.md](docs/METHODS.md). Add `--all-backbones` to exercise every backbone. Logs and a pass/fail summary are saved under `results/smoke/`.

## Run an experiment

```bash
# Default: Scaffold-Sample, asynchronous refresh, full-graph evaluation.
python run.py --dataset citeseer --workers auto

# Dataset-specific single-support and K-support settings.
python run.py --dataset citeseer --recipe scaffold-1
python run.py --dataset citeseer --recipe scaffold-k

# Full-evaluation settings; Fast and Batch have recipes too.
python run.py --dataset citeseer --method scaffold-fast --recipe scaffold-full

# Cora reference settings: one fixed support.
python run.py --dataset cora --method scaffold-greedy --recipe scaffold-1
python run.py --dataset cora --method scaffold-heap --recipe scaffold-1

# Same dataset's shared GCN settings for the full-graph comparator.
python run.py --method full --dataset cora

# Inspect every resolved setting without downloading data or starting training.
python run.py --method scaffold-batch --dataset ogbn-products --dry-run
```

`--recipe` selects the dataset's backbone, objective, seed, training schedule, and evaluation policy, including its refresh mode. Without a recipe, Fast, Batch, and Sample default to asynchronous refresh every epoch and full-graph evaluation; use `--mode single` for one fixed support or `--mode multi --views 5` for union inference. `--synchronous` waits for each refresh. The no-argument example dataset is CiteSeer.

`--epochs`, `--runs`, `--seed`, and `--ratio` override the selected settings. A union can exceed the per-support edge budget; a complete spanning forest needs at least `n − c` edges for `c` components. For a custom single/multi run, `--include-full-eval` additionally records validation-selected full-graph inference in `multiview.csv`. The `full` comparator trains and evaluates on the full graph.

## Choose an algorithm

| Method | Construction | Intended use |
|---|---|---|
| `scaffold-greedy` | Recompute scores after each insertion | Small graphs; reference |
| `scaffold-heap` | Cache paths and refresh affected candidates | Small graphs |
| `scaffold-batch` | Score candidate batches; insert their top edges | Medium/large graphs |
| `scaffold-fast` | Score on the initial forest, then select edges | Medium/large graphs |
| `scaffold-sample` | Precompute weights across forests; draw supports | Medium/large graphs; frequent refresh |

Batch defaults to 512 candidates and 64 insertions per cluster round; change them with `--batch-size` and `--add-per-round`. Fast and Batch distribute cluster work across CPU workers. Sample's `--sample-forests` controls weight precomputation, independently of the inference `--views` count.

<p align="center"><img src="docs/images/variants.png" width="780" alt="Input graph and the five Scaffold variants."></p>

## Data, settings, and results

Datasets download to `./data`; runs and caches are stored in `./results`. Relative paths are resolved from the repository root in an editable checkout, or from the working directory after installing a wheel. For existing datasets:

```bash
export SCAFFOLD_DATA_ROOT=/path/to/existing/datasets
python run.py --method scaffold-fast --dataset reddit --workers 32
```

Each run writes `resolved.json`, `run.log`, and `status.json` in a unique results directory. Native metric files and shared `metrics.csv` are preserved where produced. Cached partitions, Sample weights, and spectral artifacts live below `results/cache`. Keep caches when measuring warm-start performance; use a fresh cache directory for cold-start measurements. Process wall time includes initialization; method-reported training time is a separate quantity.

- [`configs/tunedgnn_presets.py`](configs/tunedgnn_presets.py): common architecture, optimizer, training schedules, and dataset profiles.
- [`configs/datasets.yaml`](configs/datasets.yaml): 19 paper datasets and retention sweeps.
- [`configs/methods/`](configs/methods): method-specific settings and launchers.
- [`docs/METHODS.md`](docs/METHODS.md): all comparison commands, implementation differences, and optional dependencies.
- [`docs/VALIDATION.md`](docs/VALIDATION.md): execution checks and their limits.

The data loader is self-contained; no EDSparse checkout or installation is required. Tuned GraphSAGE uses the shared **GraphSAGE** preset; Tuned GraphSAINT-RW uses the shared GCN architecture with random-walk sampling. Learned-baseline adapters and their larger-graph approximations are identified in the method documentation.

## Support backbones

All backbones return a spanning **forest** on disconnected inputs. A tree is the connected special case. `fast-randsf` is the default randomized Kruskal backbone; it does not sample uniformly from all spanning trees.

| Name | Meaning |
|---|---|
| `randsf`, `fast-randsf` | Randomized / strided randomized spanning forest |
| `maxsf`, `minsf` | Maximum-/minimum-weight spanning forest |
| `fast-maxsf`, `fast-minsf` | Bucketed approximate weighted forest |
| `randspt` (`randspf`) | Random-root shortest-path forest |
| `glsf` | Greedy low-stretch forest; slow |
| `slsf` | Scalable low-stretch heuristic |
| `llsf` | Local-search low-stretch forest; slow |

Legacy names such as `randst`, `maxst`, `mst`, `glst`, `slst`, and `llst` remain accepted. Use Greedy/Heap and low-stretch search backbones only on small graphs. The visualization below is unweighted; weighted backbones may produce different supports when supplied with edge weights.

Fast and Batch select a compatible construction backend automatically: tensor for Kruskal forests and NetworkX for shortest-path/low-stretch forests. LLSF search can be limited with `--llsf-passes`, `--llsf-candidates`, `--llsf-eval-edges`, and `--llsf-cycle-edges`; `--llsf-init randsf` selects its initial forest. The backbone smoke check uses one LLSF pass with sampled candidates and stretch evaluation.

Feature-derived weights can be selected with `--edge-weights cosine` (also `euclidean` or `dot`); `--weighted-paths` additionally enables weighted supporting-path lengths. Uniform weights and hop-count paths are the default.

<p align="center"><img src="docs/images/backbones.png" width="850" alt="Spanning-forest backbones and expanded method names on an unweighted graph."></p>

<details><summary>Supporting paths, dilation, and congestion</summary>
<img src="docs/images/quantities.png" width="850" alt="Supporting paths represent omitted edges; dilation measures path length and congestion measures shared edges or internal vertices.">
</details>

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for experiment protocols and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.
