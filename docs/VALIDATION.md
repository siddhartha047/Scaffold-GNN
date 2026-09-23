# Execution checks

Cora GPU checks completed on 2026-09-23: **44 configurations passed**, each with five training epochs and one run. The graph has 2,708 nodes and 5,278 undirected edges. Checks used A100 80GB GPUs and four construction workers. The machine-readable record is [validation/cora_smoke.json](validation/cora_smoke.json).

## Coverage

| Check | Result |
|---|---|
| Scaffold Greedy, Heap, Batch, Fast, Sample | All five passed |
| Full Graph and No Graph | Both passed |
| Random, spanning forest, local degree, rank degree, Forest Fire | All passed |
| SCAN, L-Spar, G-Spar, L-Sim, Spectral, t-spanner | All passed; Spectral includes Julia effective-resistance precomputation |
| DSpar, MoG, AdaGLT, Unified-LTH | All passed |
| Tuned GraphSAGE and Tuned GraphSAINT-RW | Both passed |
| Ten support backbones with Scaffold-Fast | All passed |
| Cora Greedy-1 and Heap-1 recipes | Both passed |
| Weighted Fast/Batch construction | Cosine/Fast-MaxSF and Euclidean/Fast-MinSF passed with weighted supporting paths |
| Fast/Batch/Sample union and full-graph inference | Best-1, union-3, and full-graph results reported `OK`; unions used three supports and full evaluation retained all 5,278 edges |
| Fast/Batch/Sample asynchronous construction | Four refreshes built and applied in each five-epoch completion check |
| Public smoke runner | Method and backbone checks completed and wrote their summaries |
| Unit/integration tests | 52 passed, including spectral artifact checks, backend selection, and all 164 recipe configurations |

The ten backbones are `fast-randsf`, `randsf`, `maxsf`, `minsf`, `fast-maxsf`, `fast-minsf`, `randspt`, `slsf`, `glsf`, and `llsf`. LLSF used a RandSF initializer, one search pass, 16 candidate edges, 128 stretch-evaluation edges, and four removable cycle edges. GLSF used its full construction.

AdaGLT and Unified-LTH smoke checks use five mask-search epochs followed by five sparse-model training epochs; Unified-LTH uses one pruning round. The dedicated asynchronous completion checks enable `--scaffold_async_wait` in the core runner so background work completes even on this short workload. Ordinary asynchronous training continues without waiting.

## Repeat the checks

Install the Python/PyG environment and [Spectral's Julia dependencies](METHODS.md#spectral-precomputation), then run:

```bash
# All 24 method entry points; automatically precompute Spectral artifacts.
python scripts/smoke.py --dataset cora --epochs 5 --all-methods --workers 4

# All ten backbones, using Scaffold-Fast with one fixed support.
python scripts/smoke.py --dataset cora --epochs 5 --methods --all-backbones --workers 4

# Unit and integration tests.
python -m pytest -q tests RelatedMethods/Spectral/tests
```

Each smoke check saves its log, exact command, exit status, and elapsed time under `results/smoke/`. A `summary.json` collects the outcomes. Use `--output PATH` to choose the report directory and `--timeout SECONDS` to set the per-check limit. GLSF and full LLSF searches can be slow; the LLSF smoke settings above bound its search work.

## Environment

Python 3.12.9; PyTorch 2.6.0+cu124; PyG 2.6.1; torch-sparse 0.6.18+pt26cu124; torch-scatter 2.1.2+pt26cu124; NumPy 2.2.6; SciPy 1.15.3; NetworKit 11.2.1; Numba 0.66.0; OGB 1.3.6; scikit-learn 1.6.1; PyYAML 6.0.2.
