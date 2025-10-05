# CirrusClassify Architecture Overview

This note captures the moving pieces of the modernised CirrusClassify project so that contributors can navigate the code base and extend it safely.

## Package layout

| Package | Responsibility | Key classes/modules |
| --- | --- | --- |
| `cirrusclassify.io` | Data ingestion from CloudSat HDF/ancillary sources, shared readers. | `cloudsat_reader.CloudSatReader` |
| `cirrusclassify.processing` | Feature engineering, classification pipelines, batch orchestration. | `batch.BatchProcessor`, `features.FeatureEngineer` |
| `cirrusclassify.analysis` | Statistical post-processing, regressions, climatology aggregations. | *(to be migrated from legacy scripts)* |
| `cirrusclassify.visualization` | Publication-quality figures and QC quick looks. | *(to be migrated from legacy scripts)* |
| `cirrusclassify.utils` | Cross-cutting helpers (geospatial bins, constants, maths). | `geo.compute_region` |

The package follows a `src/` layout and is installable via `pip install -e .`; submodules should avoid hard-coded relative paths so that they can be reused from notebooks or external projects.

## Data flow (current pipeline)

1. **Ingestion** – `CloudSatReader` wraps the HDF4 interfaces (`pyhdf.HDF`, `pyhdf.SD`), applies scaling/masking metadata, and yields numpy arrays for geolocation, time, and science datasets.
2. **Feature engineering** – `FeatureEngineer` converts arrays to tabular features (latitude/longitude, banded regions, mask values, elevation). This module is intentionally lightweight and should remain composable.
3. **Batch orchestration** – `BatchProcessor` walks CloudSat directories, instantiates readers/engineers, and persists intermediate outputs (currently Parquet). It is configured from YAML files under `configs/` and is the entry point for CLI scripts.
4. **Downstream products** *(in progress)* – Legacy scripts provide advanced classification, morphological operations, and climatology notebooks. As functionality is ported, create reusable modules under `processing`, `analysis`, or `visualization` and write thin wrappers in `scripts/`.

## Configuration conventions

- Runtime settings live in YAML files (`configs/`). Each top-level dictionary maps to a component (e.g., `data`, `features`, `output`).
- CLI wrappers (such as `scripts/run_pipeline.py`) accept the path to a YAML file, deserialize it, and hand it to `BatchProcessor.from_config`.
- Outputs default to `./outputs` (created automatically). Downstream consumers should treat the directory as disposable artefacts that can be regenerated.

## Migration roadmap

The historical `muqy_*.py` scripts encode the full scientific workflow: connected-component cirrus classification, monthly climatologies, aerosol filtering, regression experiments, and plotting. The plan is to incrementally:

1. Identify reusable algorithms within each script and lift them into `src/cirrusclassify/` modules with clean APIs.
2. Provide tests under `tests/` to lock in behaviour (especially scaling/masking, morphological operations, and statistics).
3. Replace monolithic scripts with thin CLI shells located in `scripts/` that wire configuration into the new modules.
4. Document each module as it lands (docstrings + Sphinx notes) and surface usage examples in the README/notebooks.

Contributors are encouraged to follow these steps when migrating new functionality to maintain consistency across the code base.
