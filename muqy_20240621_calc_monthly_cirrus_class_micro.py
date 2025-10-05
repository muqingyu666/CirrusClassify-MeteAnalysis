# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-06-21 08:45
# @Last Modified by:   Muqy
# @Last Modified time: 2024-09-01 22:53


import gc
import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from muqy_20240312_generate_cirrus_class_grid import extract_start_date

########################################################################################
#### Functions for calculate monthly averaged cirrus microphysical properties ##########
########################################################################################


def group_files_by_month(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Groups a list of file paths by month.

    Args:
        file_paths (List[str]): A list of file paths.

    Returns:
        Dict[str, List[str]]: A dictionary where the keys are the month in the format "YYYY-MM"
        and the values are lists of file paths that belong to that month.
    """
    file_paths_by_month = {}
    for file_path in file_paths:
        start_date = extract_start_date(Path(file_path).name)
        month = start_date.strftime("%Y-%m")
        file_paths_by_month.setdefault(month, []).append(file_path)
    return file_paths_by_month


def process_file(file_path, variables, grid_shape):
    """
    Processes a single CloudSat file and calculates sums and counts for specified variables.

    Args:
        file_path (str): Path to the CloudSat data file.
        variables (list): List of variables to process.
        grid_shape (tuple): Shape of the grid for interpolation.
    Returns:
        dict: Sums and counts for each variable.
    """
    with xr.open_dataset(file_path, engine="h5netcdf") as ds:
        # Latitude and longitude
        lats, lons = ds["latitude"].values, ds["longitude"].values

        # Cloud masks
        insitu_mask = ds["insitu_mask"].values
        anvil_mask = ds["anvil_mask"].values

        lat_bins = np.linspace(-90, 90, grid_shape[1] + 1)
        lon_bins = np.linspace(-180, 180, grid_shape[2] + 1)

        results = {
            f"{var}_insitu": {
                "sum": np.zeros(grid_shape),
                "count": np.zeros(grid_shape),
            }
            for var in variables
        }
        results.update(
            {
                f"{var}_anvil": {
                    "sum": np.zeros(grid_shape),
                    "count": np.zeros(grid_shape),
                }
                for var in variables
            }
        )

        # Loop over all nray
        for i in range(lats.shape[0]):
            # Find the lat and lon indices for the current nray
            lat_ind = np.digitize(lats[i], lat_bins) - 1
            lon_ind = np.digitize(lons[i], lon_bins) - 1

            # Check if the lat and lon indices are within the grid shape
            if (
                0 <= lat_ind < grid_shape[1]
                and 0 <= lon_ind < grid_shape[2]
            ):
                # Loop over height bins
                for h in range(grid_shape[0]):
                    insitu_mask_current = insitu_mask[i, h]
                    anvil_mask_current = anvil_mask[i, h]

                    if insitu_mask_current == 1:
                        for var in variables:
                            data = ds[var].values[i, h]
                            if not np.isnan(data):
                                results[f"{var}_insitu"]["sum"][
                                    h, lat_ind, lon_ind
                                ] += data
                                results[f"{var}_insitu"]["count"][
                                    h, lat_ind, lon_ind
                                ] += 1

                    elif anvil_mask_current == 1:
                        for var in variables:
                            data = ds[var].values[i, h]
                            if not np.isnan(data):
                                results[f"{var}_anvil"]["sum"][
                                    h, lat_ind, lon_ind
                                ] += data
                                results[f"{var}_anvil"]["count"][
                                    h, lat_ind, lon_ind
                                ] += 1

    # Clear memory
    gc.collect()
    return results


def calculate_monthly_averages(
    file_paths, variables, grid_shape=(125, 90, 180), num_jobs=-1
):
    """
    Calculate monthly averages for specified variables on a uniform grid.

    Args:
        file_paths (list): List of file paths containing CloudSat data.
        variables (list): List of variables to process.
        grid_shape (tuple): Shape of the grid for interpolation.
        num_jobs (int): Number of parallel jobs to run.

    Returns:
        dict: Monthly averages for each variable.
    """
    file_paths_by_month = group_files_by_month(file_paths)
    monthly_results = {}

    for month, paths in file_paths_by_month.items():
        print(f"Processing month: {month}")
        results = Parallel(n_jobs=num_jobs, backend="loky")(
            delayed(process_file)(fp, variables, grid_shape)
            for fp in paths
        )

        combined_results = {
            f"{var}_insitu": {
                "sum": np.zeros(grid_shape),
                "count": np.zeros(grid_shape),
            }
            for var in variables
        }
        combined_results.update(
            {
                f"{var}_anvil": {
                    "sum": np.zeros(grid_shape),
                    "count": np.zeros(grid_shape),
                }
                for var in variables
            }
        )

        for res in results:
            for var in variables:
                combined_results[f"{var}_insitu"]["sum"] += res[
                    f"{var}_insitu"
                ]["sum"]
                combined_results[f"{var}_insitu"]["count"] += res[
                    f"{var}_insitu"
                ]["count"]
                combined_results[f"{var}_anvil"]["sum"] += res[
                    f"{var}_anvil"
                ]["sum"]
                combined_results[f"{var}_anvil"]["count"] += res[
                    f"{var}_anvil"
                ]["count"]

        monthly_averages = {
            f"{var}_insitu": np.divide(
                combined_results[f"{var}_insitu"]["sum"],
                combined_results[f"{var}_insitu"]["count"],
                out=np.full(grid_shape, np.nan),
                where=combined_results[f"{var}_insitu"]["count"] != 0,
            )
            for var in variables
        }
        monthly_averages.update(
            {
                f"{var}_anvil": np.divide(
                    combined_results[f"{var}_anvil"]["sum"],
                    combined_results[f"{var}_anvil"]["count"],
                    out=np.full(grid_shape, np.nan),
                    where=combined_results[f"{var}_anvil"]["count"]
                    != 0,
                )
                for var in variables
            }
        )

        monthly_results[month] = monthly_averages
        print(f"{month}: Done")

    return monthly_results


def save_results(results, heights, output_dir, variables):
    """
    Save results to netCDF files using h5netcdf engine.

    Args:
        results (dict): Monthly averages for each variable.
        output_dir (str): Directory to save the output files.
        variables (list): List of variables processed.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = (
        Path(output_dir)
        / "CloudSat_cirrus_class_micro_monthly_mean_aerosol_0085.nc"
    )

    # Sort months to ensure chronological order
    sorted_months = sorted(results.keys())

    # Create time coordinate
    times = pd.to_datetime(sorted_months, format="%Y-%m")

    # Get dimensions from the first month's data
    first_month_data = results[sorted_months[0]]

    latitudes = np.linspace(
        -90, 90, first_month_data[variables[0]].shape[1]
    )
    longitudes = np.linspace(
        -180, 180, first_month_data[variables[0]].shape[2]
    )

    # Prepare data arrays for each variable
    data_arrays = {}

    for var in variables:
        # Stack data for all months
        stacked_data = np.stack(
            [results[month][var] for month in sorted_months]
        )

        data_arrays[var] = xr.DataArray(
            stacked_data.astype(np.float32),
            dims=("time", "height", "lat", "lon"),
            coords={
                "time": times,
                "height": heights,
                "lat": latitudes,
                "lon": longitudes,
            },
        )

    # Create dataset
    ds = xr.Dataset(data_arrays)

    # Set encoding for compression
    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds.data_vars
    }

    # Save to netCDF
    ds.to_netcdf(
        output_file, mode="w", engine="h5netcdf", encoding=encoding
    )
    print(f"Saved all monthly averages to {output_file}")


########################################################################################

if __name__ == "__main__":

    pass
