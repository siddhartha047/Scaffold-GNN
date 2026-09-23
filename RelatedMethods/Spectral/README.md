# Effective-resistance precomputation

Julia JL/approximate-Cholesky and C++ PCG/TGT backends are included. Use the root `scripts/precompute_spectral.py` interface to keep output artifacts in `results/cache/spectral`. See [installation and commands](../../docs/METHODS.md#spectral-precomputation).

```bash
julia --project=RelatedMethods/Spectral -e 'using Pkg; Pkg.instantiate()'
python scripts/precompute_spectral.py --dataset cora --workers 8
python run.py --method spectral --dataset cora --ratio 0.7
```

The top-level runner uses exact-budget spectral sampling. The standalone `run.py` in this directory demonstrates the original with-replacement sampler; its draw count is not its number of unique retained edges. Small-graph numerical tests are in `tests/` and `test/`.
