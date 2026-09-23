# Validation record

Checked on 2026-09-23. These are execution and integration checks, not a full accuracy campaign.

## Completed checks

| Check | Result |
|---|---|
| Unit/integration tests, including spectral artifact tests | 24 passed |
| Shared configuration matrix | All 24 methods × 19 paper datasets resolve; core argument lists parse |
| GPU smoke matrix | All 24 method entry points completed one tiny Karate training run |
| Cora GPU smoke runs | Full Graph; Fast, Batch, Sample; DSpar, MoG, AdaGLT, Unified-LTH; Tuned GraphSAGE; Tuned GraphSAINT-RW passed |
| Parallel construction | Fast and Batch completed Cora with four partitions and four workers actually used |
| Weighted construction | Fast-MaxSF, cosine weights, and weighted supporting paths completed a Cora GPU run |
| Multiple-support inference | Sample refreshed synchronously for five tiny epochs; best-1 and union-3 rows both reported `OK`, with three effective supports |
| Julia spectral path | Computed approximate effective resistances on Karate; the top-level exact-budget Spectral consumer then trained successfully on GPU |
| Large-graph C++ backend | Built with CMake; JL-PCG completed a small graph with 16 projections and two threads |
| Larger-graph MoG module integration | Both Arxiv and Proteins native learner modules loaded locally and passed tiny forward/backward checks |
| Packaging | Built a wheel and ran an isolated installed-wheel Fast training smoke test |
| Anonymous archive | Verified exclusions and required source files; extracted the archive separately and completed a CPU Fast training run |

GPU checks used two existing A100 80GB sessions, each constrained to its assigned visible GPU and 32-core CPU affinity. No Slurm submission or new allocation was requested. Tiny tests used two workers; explicit parallel Cora tests used four. Original research source files were left in place.

## Environment

Python 3.12.9; PyTorch 2.6.0+cu124; PyG 2.6.1; torch-sparse 0.6.18+pt26cu124; torch-scatter 2.1.2+pt26cu124; NumPy 2.2.6; SciPy 1.15.3; NetworKit 11.2.1; Numba 0.66.0; OGB 1.3.6; scikit-learn 1.6.1; PyYAML 6.0.2. DGL 2.4.0+cu124 imported successfully. Julia dependencies were available from the supplied spectral project.

## Re-run

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests RelatedMethods/Spectral/tests
python scripts/smoke.py --methods full scaffold-greedy scaffold-heap scaffold-batch scaffold-fast scaffold-sample
python scripts/smoke.py --methods dspar mog adaglt unified-lth tuned-graphsage tuned-graphsaint-rw
```

Spectral smoke training needs an artifact first. For a self-contained tiny example:

```bash
python RelatedMethods/Spectral/run.py --dataset karate --threads 2 \
  --report results/spectral-karate.csv
python run.py --method spectral --dataset karate --smoke
```

Raw smoke outputs remain in the local ignored `results/` directory. The anonymous archive contains this summary, not machine paths, logs, checkpoints, or results.

## Limits

Full-size training and accuracy across all 19 datasets have not been rerun. The configuration matrix checks coverage and argument validity, not graph download availability or memory fit. The default asynchronous refresh path is inherited; the completed tiny multiview test used synchronous refresh to ensure three supports were constructed before training ended. The larger-graph AdaGLT/Unified-LTH accuracy adapters remain different from native learned-mask methods, as documented in `METHODS.md`. Reported-accuracy settings remain pending.
