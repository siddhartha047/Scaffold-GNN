# Third-party code and attribution

This distribution preserves comparison implementations as separate folders. Adaptations cover shared dataset loading, TunedGNN model settings, portable paths, result recording, retained-edge budgets, and the documented larger-graph execution paths. They are not represented as newly authored baseline algorithms.

| Directory | Source / role | Included notice |
|---|---|---|
| `RelatedMethods/tunedGNN-main` | TunedGNN model/training code; upstream revision recorded in `configs/tunedgnn_presets.py` | Original MIT `LICENSE` |
| `RelatedMethods/Unified-LTH-GNN-main` | Unified lottery-ticket graph pruning | Original MIT `LICENSE` |
| `RelatedMethods/AdaGLT-main` | Adaptive graph lottery-ticket GCN code | No standalone license file was present in the supplied source folder |
| `RelatedMethods/MoG-main` | Mixture-of-graphs expert/learner implementations | No standalone license file was present in the supplied source folder |
| `RelatedMethods/DSpar_tmlr-main` | DSpar degree-based graph sparsification | No standalone license file was present in the supplied source folder |
| `RelatedMethods/GraphSAINT-RW` | PyG GraphSAINT random-walk integration | Relies on separately installed PyG |
| `RelatedMethods/TunedGNN-GraphSAINT-RW` | GraphSAINT integration with the TunedGNN comparison model | TunedGNN attribution above |
| `RelatedMethods/Spectral` | Effective-resistance precomputation integration | Julia/Python/C++ dependencies retain their own licenses |

Original copyright/license notices are retained wherever supplied. This cleanup does not grant a blanket license over third-party code or replace those terms.

`docs/source_manifest.json` records hashes of source files at import time. Destination files were subsequently adapted; those hashes identify the imported snapshots, not checksums of the final transformed files. A destination entry may refer to a renamed or pruned source file.
