# CirrusClassify-MeteAnalysis

**CirrusClassify-MeteAnalysis** is a research-grade Python package for analysing CloudSat observations, isolating anvil versus in-situ cirrus, and producing downstream diagnostics such as climatologies, microphysical statistics, and machine-learning ready features. The project now follows a modern `src/` layout with reusable modules, configuration-driven pipelines, documentation, and automated testing hooks.

## 🧱 Project structure

```
.
├── src/cirrusclassify/           # Installable Python package
│   ├── io/                       # CloudSat readers & ancillary loaders
│   ├── processing/               # Feature engineering & batch pipelines
│   ├── analysis/                 # Statistical and ML utilities
│   ├── visualization/            # Plotting helpers
│   └── utils/                    # Shared math/geo helpers
├── configs/                      # YAML configuration templates
├── scripts/                      # Runnable entry points for pipelines & legacy glue
├── tests/                        # pytest-based unit tests
├── docs/                         # Architecture notes & Sphinx-ready docs
├── data/                         # (Optional) local cache, ignored by git
├── notebooks/                    # Exploratory notebooks
├── pyproject.toml                # Packaging & dependency metadata
├── requirements.txt              # Runtime dependency pin list (pip install -r)
├── Makefile                      # Common developer tasks (setup, lint, test)
└── README.md                     # You are here
```

Legacy one-off scripts remain at the repository root for the moment. They will be gradually migrated into the package modules; feel free to call them directly while transition work continues.

## 🚀 Quick start

1. **Create an environment** (Python 3.10+ recommended) and install the package in editable mode:

```bash
make setup
```

2. **Run unit tests** to verify the installation:

```bash
make unit
```

3. **Execute the classification pipeline** using the new orchestration script:

```bash
python scripts/run_pipeline.py configs/pipeline.defaults.yaml
```

Edit the YAML file to point at your CloudSat 2B-GEOPROF granules and to configure output directories or feature settings. The default pipeline:

- opens each HDF granule with `cirrusclassify.io.CloudSatReader`
- derives location-aware features via `cirrusclassify.processing.FeatureEngineer`
- writes per-granule Parquet tables containing mask/feature columns

4. **Inspect outputs** – the default configuration saves derived features under `./outputs`. Downstream notebooks in `notebooks/` or scripts in `scripts/` can consume these tables for climatology, regression, or visual QC.

## 🧩 Core modules

- `cirrusclassify.io.cloudsat_reader.CloudSatReader`: context-manager friendly HDF4 interface that applies scaling/masking metadata and returns geolocation, time, and SDS arrays.
- `cirrusclassify.processing.batch.BatchProcessor`: configuration-driven orchestrator that loops over CloudSat granules, triggers feature extraction, and writes results.
- `cirrusclassify.processing.features.FeatureEngineer`: example feature pipeline producing geospatial metadata and mask summaries; extend or subclass for project-specific metrics.
- `cirrusclassify.utils.geo`: helper functions for geographical binning (latitudinal bands, etc.).

Additional subpackages (`analysis`, `visualization`) will be populated as legacy scripts migrate into reusable modules.

## ⚙️ Configuration

YAML templates in `configs/` define runtime behaviour. The bundled `pipeline.defaults.yaml` contains:

```yaml
project:
  name: cirrusclassify
data:
  cloudsat_dir: "/data/cloudsat/geoprof"
  primary_sds: "Cloud_mask"
features:
  primary_variable: "Cloud_mask"
output:
  directory: "./outputs"
logging:
  level: INFO
```

Override any path or parameter (e.g., target SDS name, output location) by creating a copy of the file and passing it to the CLI. Nested dictionaries map directly to the configuration consumed by `BatchProcessor` and downstream utilities.

## 🧪 Testing & quality

- `pytest` drives unit tests under `tests/`.
- `ruff` and `black` enforce style; run `make lint` or `make format`.
- Coverage configuration is set up in `pyproject.toml` (`pytest --cov`).

These commands are safe to execute locally and in CI pipelines.

## 🗺️ Legacy scripts & notebooks

The historical analysis scripts (prefixed `muqy_...`) and notebooks document the full range of research experiments—plotting, climatology aggregation, machine learning, and case studies. They continue to function with the new package and serve as references for upcoming refactors. When porting functionality, prefer creating reusable modules under `src/cirrusclassify/` and thin wrappers in `scripts/`.

## 📚 Documentation

- `docs/architecture.md` captures high-level design notes and the migration plan.
- Sphinx configuration is pre-wired; run `make docs` to build HTML documentation once docstrings and `.rst`/Markdown sources are ready.

## � Dependencies

Key runtime libraries (managed via `pyproject.toml`): `numpy`, `pandas`, `xarray`, `matplotlib`, `cartopy`, `seaborn`, `scipy`, `joblib`, `pyhdf`, `metpy`, `scikit-learn`, `contextily`, `h5netcdf`, `pyarrow`. Install HDF4 support (`brew install hdf4` or Conda equivalent) before installing `pyhdf` on macOS.

Optional developer extras (`pip install .[develop]`:

- Tooling: `black`, `ruff`, `pytest`, `pytest-cov`, `pre-commit`
- Docs: `sphinx`, `myst-parser`

## 📜 Licence & citation

Distributed under the terms of the project `LICENSE` (MIT). When referencing this work, please cite: *Qingyu Mu, CirrusClassify-MeteAnalysis, 2024–2025.*

Questions, feature requests, or bug reports are welcome via repository issues or contact details in file headers.
