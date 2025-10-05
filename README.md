# CirrusClassify-MeteAnalysis

CirrusClassify-MeteAnalysis is a script-driven toolkit for processing CloudSat observations with a focus on separating anvil and in-situ cirrus, extracting microphysical properties, aggregating statistics, and producing diagnostic visualisations. The code base covers raw HDF ingestion, connected-component labelling, large-scale parallel processing, fusion with ECMWF/MODIS auxiliary data, and downstream statistical or machine-learning analyses.

## 🔁 Overview & end-to-end pipeline

1. **Ingestion** – `muqy_20240101_2Bgeoprof_reader.py` exposes a unified reader for CloudSat R05 2B HDF files, automatically handling Vdata/SDS scaling and masking.
2. **Per-track QC plots** – `muqy_20240101_plot_2Bgeoprof_test.py` and `muqy_20240102_plot_2Bcldtype_2Bcldfrac_2Cice.py` provide longitudinal cross-section plots and orbit tracks for CLDCLASS-LIDAR, 2C-ICE, and 2B-GEOPROF products.
3. **Cirrus classification pipeline** – `muqy_20240104_filter_anvil_insitu_cirrus.py` implements `CloudSatProcessor_Ver2`, combining connected-component labelling, morphological dilation, and IWC/temperature constraints to distinguish **anvil cirrus** from **in-situ cirrus**, with helper plotting utilities.
4. **Batch parallel processing** – `muqy_20240611_apply_filter_cld_parallel_with_ice_modified.py` orchestrates multi-process classification for large collections of CloudSat granules and writes compressed NetCDF outputs containing masks, microphysics, and ECMWF auxiliaries.
5. **Post-processing & reanalysis** – follow-on scripts include:
	 - `muqy_20240619_filter_high_AOD_cirrus_zone.py`: masks cirrus scenes in regions with elevated aerosol optical depth (AOD).
	 - `muqy_20240621_calc_monthly_cirrus_class_micro.py`: aggregates masks and microphysics into monthly gridded means.
	 - `muqy_20240625_calc_aux_trop_vars_from_ogdata_og.py`: derives tropospheric metrics (tropopause height, static stability, relative humidity, etc.).
	 - `muqy_20240909_calc_height_weighted_cirrus_mask.py`: projects 3-D masks into height-weighted 2-D occurrence fields.
	 - `muqy_20240910_generate_Cirrus_inc_prof_EC_AUX_vars_mean_parallel.py`: bins cirrus occurrence and profiles by ECMWF auxiliary-variable deciles.
6. **Statistical analysis & visualisation** – scripts such as `muqy_20240923_analyze_cirrus_class_freq_micro.py`, `muqy_20241031_generate_monthly_meteo.py`, and `muqy_20250926_dbz_cloudsat_differ.py`, plus the notebook `muqy_20250423_spatial_corr_anvil_insitu.ipynb`, render global climatologies, radar reflectivity distributions, correlations, and regressions.
7. **Advanced studies** – `muqy_20241104_ridge_regression_enhanced_grid_point_test.py` performs grid-point ridge regression with cross-validation; `muqy_20250322_anvilDCS_length_calc.py` estimates contiguous anvil/deep-convection segment lengths; `muqy_20250927_reflectivity_distribution.py` compiles probability-density statistics for masked radar reflectivity.

## 📁 Repository quick reference

| File | Purpose |
| --- | --- |
| `muqy_20240101_2Bgeoprof_reader.py` | PyHDF-based reader for CloudSat R05 products (geo/time and SDS scaling/masking).
| `muqy_20240101_plot_2Bgeoprof_test.py` | Plot utilities for map setup, track rendering, and radar–lidar cross-sections.
| `muqy_20240102_plot_2Bcldtype_2Bcldfrac_2Cice.py` | Multi-product visualisation combining cloud type, fraction, ice microphysics, and orbit maps.
| `muqy_20240104_filter_anvil_insitu_cirrus.py` | Core classification workflow with connected-component labelling, morphology, and diagnostics.
| `muqy_20240611_apply_filter_cld_parallel_with_ice_modified.py` | Directory walker that processes daily batches in parallel and writes compressed NetCDF summaries.
| `muqy_20240619_filter_high_AOD_cirrus_zone.py` | Filters cirrus occurrences using MODIS-based AOD thresholds.
| `muqy_20240621_calc_monthly_cirrus_class_micro.py` | Aggregates processed NetCDF files into monthly fixed-grid cirrus occurrence and microphysics.
| `muqy_20240625_calc_aux_trop_vars_from_ogdata_og.py` | Computes thermodynamic/dynamic derivatives (potential temperature, wind shear, tropopause metrics) and saves NetCDF.
| `muqy_20240909_calc_height_weighted_cirrus_mask.py` | Converts 3-D masks into height-weighted 2-D climatologies.
| `muqy_20240910_generate_Cirrus_inc_prof_EC_AUX_vars_mean_parallel.py` | Builds decile-conditioned cirrus profiles/incidence using ECMWF auxiliaries.
| `muqy_20240923_analyze_cirrus_class_freq_micro.py` | Leverages helper modules for global dual-hemisphere maps and vertical profile figures.
| `muqy_20241031_generate_monthly_meteo.py` | Produces monthly gridded meteorology matched to height-weighted masks.
| `muqy_20241104_ridge_regression_enhanced_grid_point_test.py` | Performs grid-point ridge regression with cross-validation diagnostics.
| `muqy_20250322_anvilDCS_length_calc.py` | Measures along-track lengths of in-situ/anvil/DCS segments and outputs 2-D masks.
| `muqy_20250423_spatial_corr_anvil_insitu.ipynb` | Jupyter notebook exploring spatial correlations between cirrus classes.
| `muqy_20250924_case_study_and_modis.py` | Case-study visualisation combining CloudSat, MODIS, and basemap context.
| `muqy_20250926_dbz_cloudsat_differ.py` | Extracts 2B-GEOPROF radar reflectivity, merges with masks, and writes NetCDF.
| `muqy_20250927_reflectivity_distribution.py` | Samples masked reflectivity to compute histograms and PDFs.

> 📓 Only headline scripts are listed above. Check each script header (author, last-modified timestamp) for additional notes or one-off utilities.

## 🧩 Runtime dependencies

- **Python** – recommended 3.10 or newer.
- **Core packages** – `numpy`, `pandas`, `xarray`, `matplotlib`, `cartopy`, `seaborn`, `scipy`, `joblib`, Python `multiprocessing`, `pyhdf`, `metpy`, `scikit-learn`, `contextily`, `h5netcdf`.
- **Optional extras** – `mpl_toolkits` (bundled with matplotlib), `pyproj` (cartopy dependency), `dask` (if you expand parallelism).
- **External data** –
	- CloudSat R05 **2B-GEOPROF / 2B-CLDCLASS-LIDAR / 2C-ICE** granules.
	- CloudSat **ECMWF_AUX** and **MODIS_AUX** ancillary products.
	- MODIS or equivalent AOD NetCDF files for `muqy_20240619_filter_high_AOD_cirrus_zone.py`.

> 💡 On macOS, install HDF4 via Homebrew or Conda before running `pip/conda install pyhdf`.

## 🚀 Typical workflows

### 1. Pre-processing & classification

```bash
# Example: iterate through directories and produce processed NetCDF files
python muqy_20240611_apply_filter_cld_parallel_with_ice_modified.py
```

Key parameters to adjust:

- Configure `base_paths` at the top of the script (CLDCLASS / GEOPROF / ICE / ECMWF_AUX directories).
- `structure_0` and `structure_1` control the horizontal/vertical extent of morphological dilation.
- `iterations` sets how many expansion passes are applied to anvil cirrus.

### 2. Mask QC and visual checks

```bash
# Quickly inspect classification results for a chosen orbit
python muqy_20240102_plot_2Bcldtype_2Bcldfrac_2Cice.py
```

Adjust `base_paths` and `file_idx` to select a specific granule; `time_range_custum` trims the shown segment.

### 3. Downstream analysis

- **AOD filtering** – `python muqy_20240619_filter_high_AOD_cirrus_zone.py`
- **Monthly climatology / gridding** – `python muqy_20240621_calc_monthly_cirrus_class_micro.py`
- **Thermo/dynamic diagnostics** – `python muqy_20240625_calc_aux_trop_vars_from_ogdata_og.py`
- **Ridge regression & statistics** – `python muqy_20241104_ridge_regression_enhanced_grid_point_test.py`

Several scripts use `joblib.Parallel` or `ProcessPoolExecutor`; tune `num_workers` / `NUM_WORKERS` to match available cores and avoid I/O bottlenecks.

## 🔗 External helper modules

The following modules are referenced but not bundled; obtain them from companion repositories or private code bases by the same author:

- `muqy_20240312_generate_cirrus_class_grid` – provides helpers such as `extract_start_date`.
- `muqy_20240710_util_cirrus_class_freq_micro` – wraps data-loading and multi-panel plotting functions.
- `EC_AUX_deciles_dict` – defines ECMWF auxiliary-variable decile ranges.

Before running, make sure these modules are on `PYTHONPATH` or copied into the repository root.

## 📜 Licence & citation

- The repository follows the terms in `LICENSE`. When citing in publications, please reference “Qingyu Mu, CirrusClassify-MeteAnalysis, 2024–2025”.
- For questions about scripts or algorithms, consult the contact details in each file header.
