# Data directories

This tree holds RealPDEBench combustion data and derived artifacts for the thesis.

| Path | Purpose |
|------|---------|
| `raw/` | Unmodified downloads from Hugging Face / other sources |
| `interim/` | Partially cleaned or subsetted tensors during audit |
| `processed/` | Training-ready arrays, normalizer stats, cached splits |
| `manifests/` | JSON/YAML inventories, checksums, operating-condition tables |

**Git policy:** Large tensors (`.h5`, `.hdf5`, `.npy`, `.npz`, `.pt`, etc.) under `raw/`, `interim/`, and `processed/` are git-ignored. Only manifests, READMEs, and `.gitkeep` files are tracked. Never commit raw or processed binary datasets.
