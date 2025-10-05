# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:46:28

import gc
import glob
import os
import datetime
from calendar import monthrange

import concurrent.futures
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

########################################################################################
##### Functions for extracting dates from filenames and saving to NetCDF files #########
########################################################################################


def extract_start_date(filename):
    """
    Extracts the start date from a filename.
    """
    date_str = "_".join(filename.split("/")[-1].split("_")[-3:])
    date_str = date_str.split(".")[0]
    return datetime.datetime.strptime(date_str, "%Y_%m_%d")


def sort_files_by_date(file_paths):
    """
    Sorts file paths based on their start dates.

    Args:
        file_paths (list): List of file paths.

    Returns:
        list: Sorted list of file paths.
    """
    sorted_files = sorted(
        file_paths, key=lambda fp: extract_start_date(fp)
    )
    return sorted_files


def process_file(file_path, grid_shape, met_variables):
    """
    Processes a single CloudSat file and calculates cloud frequencies and meteorological variables.

    Args:
        file_path (str): Path to the CloudSat data file.
        grid_shape (tuple): Shape of the grid for interpolation.
        met_variables (list): List of meteorological variable names to process.

    Returns:
        dict: A dictionary containing the accumulated grids for counts and meteorological variables.
    """

    # ---------------------------------------------------------------------
    # Open the CloudSat file using xarray
    with xr.open_dataset(file_path) as ds:

        # Geo-locations
        lats, lons = ds["latitude"].values, ds["longitude"].values

        # Cloud masks
        insitu_mask = ds["insitu_fraction_weighted_2D"].values
        anvil_mask = ds["anvil_fraction_weighted_2D"].values

        # Cloud microphysics
        IWC_insitu = ds["IWC_insitu_weighted_2D"].values
        IWC_anvil = ds["IWC_anvil_weighted_2D"].values
        re_insitu = ds["re_insitu_weighted_2D"].values
        re_anvil = ds["re_anvil_weighted_2D"].values

        # Meteorological variables
        # Ensure all variables are loaded into memory
        met_data = {var: ds[var].values for var in met_variables}

    lat_bins = np.linspace(-90, 90, grid_shape[0] + 1)
    lon_bins = np.linspace(-180, 180, grid_shape[1] + 1)

    lat_indices = np.digitize(lats, lat_bins) - 1
    lon_indices = np.digitize(lons, lon_bins) - 1

    valid_indices = (
        (lat_indices >= 0)
        & (lat_indices < grid_shape[0])
        & (lon_indices >= 0)
        & (lon_indices < grid_shape[1])
    )

    # ---------------------------------------------------------------------
    # Initialize count grids
    track_count = np.zeros(grid_shape, dtype=np.int32)
    insitu_count = np.zeros(grid_shape, dtype=np.float32)
    anvil_count = np.zeros(grid_shape, dtype=np.float32)
    re_insitu_count = np.zeros(grid_shape, dtype=np.float32)
    re_anvil_count = np.zeros(grid_shape, dtype=np.float32)
    IWC_insitu_count = np.zeros(grid_shape, dtype=np.float32)
    IWC_anvil_count = np.zeros(grid_shape, dtype=np.float32)

    # Initialize grids for meteorological variables (sum and count)
    met_sums = {
        var: np.zeros(grid_shape, dtype=np.float64)
        for var in met_variables
    }
    met_counts = {
        var: np.zeros(grid_shape, dtype=np.int32)
        for var in met_variables
    }

    # ---------------------------------------------------------------------
    # Accumulate data
    valid_data = ~(np.isnan(insitu_mask) | np.isnan(anvil_mask))
    valid_mask = valid_data & valid_indices
    # Track counts
    np.add.at(
        track_count,
        (lat_indices[valid_mask], lon_indices[valid_mask]),
        1,
    )

    # In-situ
    # Cloud masks
    np.add.at(
        insitu_count,
        (lat_indices[valid_mask], lon_indices[valid_mask]),
        insitu_mask[valid_mask],
    )
    insitu_count = np.where(np.isnan(insitu_count), 0, insitu_count)
    # Cloud microphysics
    # Re
    np.add.at(
        re_insitu_count,
        (lat_indices[valid_mask], lon_indices[valid_mask]),
        re_insitu[valid_mask],
    )
    re_insitu_count = np.where(
        np.isnan(re_insitu_count),
        np.nanmean(re_insitu_count),
        re_insitu_count,
    )
    # IWC
    np.add.at(
        IWC_insitu_count,
        (lat_indices[valid_mask], lon_indices[valid_mask]),
        IWC_insitu[valid_mask],
    )
    IWC_insitu_count = np.where(
        np.isnan(IWC_insitu_count),
        np.nanmean(IWC_insitu_count),
        IWC_insitu_count,
    )
    # Anvil
    # Cloud masks
    np.add.at(
        anvil_count,
        (lat_indices[valid_mask], lon_indices[valid_mask]),
        anvil_mask[valid_mask],
    )
    anvil_count = np.where(np.isnan(anvil_count), 0, anvil_count)
    # Cloud microphysics
    # Re
    np.add.at(
        re_anvil_count,
        (lat_indices[valid_mask], lon_indices[valid_mask]),
        re_anvil[valid_mask],
    )
    re_anvil_count = np.where(
        np.isnan(re_anvil_count),
        np.nanmean(re_anvil_count),
        re_anvil_count,
    )
    # IWC
    np.add.at(
        IWC_anvil_count,
        (lat_indices[valid_mask], lon_indices[valid_mask]),
        IWC_anvil[valid_mask],
    )
    IWC_anvil_count = np.where(
        np.isnan(IWC_anvil_count),
        np.nanmean(IWC_anvil_count),
        IWC_anvil_count,
    )

    # Meteorological variables
    for var in met_variables:
        data = met_data[var]

        # Meteorological variables
        np.add.at(
            met_sums[var],
            (lat_indices[valid_mask], lon_indices[valid_mask]),
            data[valid_mask],
        )
        met_counts[var] = np.where(
            np.isnan(met_counts[var]), 0, met_counts[var]
        )

    # ---------------------------------------------------------------------
    # Return all accumulated data
    return {
        "track_count": track_count,
        "insitu_count": insitu_count,
        "re_insitu_count": re_insitu_count,
        "IWC_insitu_count": IWC_insitu_count,
        "anvil_count": anvil_count,
        "re_anvil_count": re_anvil_count,
        "IWC_anvil_count": IWC_anvil_count,
        "met_sums": met_sums,
    }


def calculate_cloud_frequency(
    file_paths, grid_shape=(90, 180), met_variables=None
):
    """
    Calculate cloud occurrence frequency and average meteorological variables.

    Args:
        file_paths (list): List of file paths containing CloudSat data.
        grid_shape (tuple): Shape of the grid for interpolation.
        num_jobs (int): Number of parallel jobs to run.
        met_variables (list): List of meteorological variable names to process.

    Returns:
        dict: A dictionary containing cloud frequencies and meteorological variables.
    """
    if not file_paths:
        print("No files selected within the specified range.")
        return None

    if met_variables is None:
        met_variables = []

    results = Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(process_file)(fp, grid_shape, met_variables)
        for fp in file_paths
    )

    # Initialize accumulators
    total_track_count = np.zeros(grid_shape, dtype=np.int32)
    total_insitu_count = np.zeros(grid_shape, dtype=np.float32)
    total_anvil_count = np.zeros(grid_shape, dtype=np.float32)
    total_re_insitu_count = np.zeros(grid_shape, dtype=np.float32)
    total_re_anvil_count = np.zeros(grid_shape, dtype=np.float32)
    total_IWC_insitu_count = np.zeros(grid_shape, dtype=np.float32)
    total_IWC_anvil_count = np.zeros(grid_shape, dtype=np.float32)

    total_met_sums = {
        var: np.zeros(grid_shape, dtype=np.float64)
        for var in met_variables
    }
    total_met_counts = {
        var: np.zeros(grid_shape, dtype=np.int32)
        for var in met_variables
    }

    # Aggregate results from all files
    for res in results:
        total_track_count += res["track_count"]
        total_insitu_count += res["insitu_count"]
        total_re_insitu_count += res["re_insitu_count"]
        total_IWC_insitu_count += res["IWC_insitu_count"]
        total_anvil_count += res["anvil_count"]
        total_re_anvil_count += res["re_anvil_count"]
        total_IWC_anvil_count += res["IWC_anvil_count"]

        for var in met_variables:
            total_met_sums[var] += res["met_sums"][var]
            total_met_counts[var] += res["track_count"]

    # Calculate frequencies
    insitu_freq = np.divide(
        total_insitu_count,
        total_track_count,
        out=np.zeros_like(total_insitu_count, dtype=np.float32),
        where=total_track_count != 0,
    )
    re_insitu_freq = np.divide(
        total_re_insitu_count,
        total_track_count,
        out=np.zeros_like(total_re_insitu_count, dtype=np.float32),
        where=total_track_count != 0,
    )
    IWC_insitu_freq = np.divide(
        total_IWC_insitu_count,
        total_track_count,
        out=np.zeros_like(total_IWC_insitu_count, dtype=np.float32),
        where=total_track_count != 0,
    )
    anvil_freq = np.divide(
        total_anvil_count,
        total_track_count,
        out=np.zeros_like(total_anvil_count, dtype=np.float32),
        where=total_track_count != 0,
    )
    re_anvil_freq = np.divide(
        total_re_anvil_count,
        total_track_count,
        out=np.zeros_like(total_re_anvil_count, dtype=np.float32),
        where=total_track_count != 0,
    )
    IWC_anvil_freq = np.divide(
        total_IWC_anvil_count,
        total_track_count,
        out=np.zeros_like(total_IWC_anvil_count, dtype=np.float32),
        where=total_track_count != 0,
    )

    # Calculate average meteorological variables
    met_averages = {}
    for var in met_variables:
        met_averages[var] = np.divide(
            total_met_sums[var],
            total_met_counts[var],
            out=np.zeros_like(total_met_sums[var], dtype=np.float32),
            where=total_met_counts[var] != 0,
        )

    return {
        "insitu_freq": insitu_freq,
        "re_insitu_freq": re_insitu_freq,
        "IWC_insitu_freq": IWC_insitu_freq,
        "anvil_freq": anvil_freq,
        "re_anvil_freq": re_anvil_freq,
        "IWC_anvil_freq": IWC_anvil_freq,
        "met_averages": met_averages,
    }


def group_files_into_monthly_intervals(sorted_file_paths):
    """
    Groups sorted file paths into monthly intervals with accurate start and end dates.

    Args:
        sorted_file_paths (list): Sorted list of file paths.

    Returns:
        list of dict: Each dict contains 'start_date', 'end_date', and 'file_paths' for each month.
    """
    intervals = []
    if not sorted_file_paths:
        return intervals

    # Initialize the first interval
    current_start = extract_start_date(sorted_file_paths[0])
    current_start = current_start.replace(
        day=1
    )  # First day of the month
    _, last_day = monthrange(
        current_start.year, current_start.month
    )  # Last day of the month
    current_end = current_start.replace(day=last_day)

    current_group = {
        "start_date": current_start,
        "end_date": current_end,
        "file_paths": [],
    }

    for fp in sorted_file_paths:
        file_start = extract_start_date(fp)
        if current_start <= file_start <= current_end:
            current_group["file_paths"].append(fp)
        else:
            # Append the completed interval
            intervals.append(current_group)
            # Start a new interval for the new month
            current_start = file_start.replace(day=1)
            _, last_day = monthrange(
                current_start.year, current_start.month
            )
            current_end = current_start.replace(day=last_day)
            current_group = {
                "start_date": current_start,
                "end_date": current_end,
                "file_paths": [fp],
            }

    # Append the last interval
    intervals.append(current_group)

    return intervals


def process_interval_concurrent(interval, grid_shape, met_variables):
    """
    Processes a single 8-day interval and calculates cloud frequencies and meteorological averages.

    Args:
        interval (dict): A dictionary containing 'start_date', 'end_date', and 'file_paths'.
        grid_shape (tuple): Shape of the grid for interpolation.
        num_jobs (int): Number of parallel jobs to run within calculate_cloud_frequency.
        met_variables (list): List of meteorological variable names to process.

    Returns:
        dict: A dictionary containing frequencies, meteorological averages, and interval times.
    """
    start_date = interval["start_date"]
    end_date = interval["end_date"]
    paths = interval["file_paths"]

    result = calculate_cloud_frequency(paths, grid_shape, met_variables)

    if result is None:
        print(
            f"No data for interval {start_date.date()} to {end_date.date()}. Skipping."
        )
        return None

    insitu_freq = result["insitu_freq"]
    re_insitu_freq = result["re_insitu_freq"]
    IWC_insitu_freq = result["IWC_insitu_freq"]
    anvil_freq = result["anvil_freq"]
    re_anvil_freq = result["re_anvil_freq"]
    IWC_anvil_freq = result["IWC_anvil_freq"]
    met_averages = result["met_averages"]

    interval_result = {
        "insitu_freq": insitu_freq,
        "re_insitu_freq": re_insitu_freq,
        "IWC_insitu_freq": IWC_insitu_freq,
        "anvil_freq": anvil_freq,
        "re_anvil_freq": re_anvil_freq,
        "IWC_anvil_freq": IWC_anvil_freq,
        "met_averages": met_averages,
        "start_date": start_date,
        "end_date": end_date,
    }

    # Clear memory
    del insitu_freq, anvil_freq, met_averages, result
    gc.collect()

    print(f"{start_date.date()} to {end_date.date()}: Done")

    return interval_result


def calculate_8day_frequencies(
    file_paths,
    grid_shape=(90, 180),
    max_workers=40,
    met_variables=None,
):
    """
    Calculate 8-day cloud occurrence frequencies and meteorological variable averages.

    Args:
        file_paths (list): List of file paths containing CloudSat data.
        grid_shape (tuple): Shape of the grid for interpolation.
        num_jobs (int): Number of parallel jobs to run.
        met_variables (list): List of meteorological variable names to process.

    Returns:
        None
    """
    if not file_paths:
        print("No files selected within the specified range.")
        return None

    if met_variables is None:
        met_variables = []

    # Group files into 8-day intervals
    intervals = group_files_into_monthly_intervals(file_paths)

    all_insitu_freq = []
    all_re_insitu_freq = []
    all_IWC_insitu_freq = []
    all_anvil_freq = []
    all_re_anvil_freq = []
    all_IWC_anvil_freq = []
    all_met_averages = {var: [] for var in met_variables}
    all_times = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Prepare futures
        futures = [
            executor.submit(
                process_interval_concurrent,
                interval,
                grid_shape,
                met_variables,
            )
            for interval in intervals
        ]

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is None:
                continue  # Skip intervals with no data

            all_insitu_freq.append(res["insitu_freq"])
            all_re_insitu_freq.append(res["re_insitu_freq"])
            all_IWC_insitu_freq.append(res["IWC_insitu_freq"])
            all_anvil_freq.append(res["anvil_freq"])
            all_re_anvil_freq.append(res["re_anvil_freq"])
            all_IWC_anvil_freq.append(res["IWC_anvil_freq"])

            for var in met_variables:
                all_met_averages[var].append(res["met_averages"][var])

            # Record the interval start and end dates
            all_times.append(pd.to_datetime(res["start_date"].date()))

    if not all_insitu_freq:
        print("No data processed. Exiting.")
        return

    # Combine all interval data into a single dataset
    combined_insitu_freq = np.stack(all_insitu_freq, axis=0)
    combined_re_insitu_freq = np.stack(all_re_insitu_freq, axis=0)
    combined_IWC_insitu_freq = np.stack(all_IWC_insitu_freq, axis=0)
    combined_anvil_freq = np.stack(all_anvil_freq, axis=0)
    combined_re_anvil_freq = np.stack(all_re_anvil_freq, axis=0)
    combined_IWC_anvil_freq = np.stack(all_IWC_anvil_freq, axis=0)
    combined_times = [pd.to_datetime(f"{t.date()}") for t in all_times]

    # Combine meteorological variables
    combined_met_averages = {}
    for var in met_variables:
        combined_met_averages[var] = np.stack(
            all_met_averages[var], axis=0
        )

    return (
        combined_insitu_freq,
        combined_re_insitu_freq,
        combined_IWC_insitu_freq,
        combined_anvil_freq,
        combined_re_anvil_freq,
        combined_IWC_anvil_freq,
        combined_met_averages,
        combined_times,
    )


def save_combined_to_netCDF(
    output_dir,
    insitu_grids,
    re_insitu_grids,
    IWC_insitu_grids,
    anvil_grids,
    re_anvil_grids,
    IWC_anvil_grids,
    met_grids,
    times,
    interval_type="monthly",
):
    """
    Saves combined grids to a NetCDF file.

    Args:
        output_dir (str): Directory to save the output file.
        insitu_grids (np.ndarray): Insitu mask grid data with shape (time, lat, lon).
        anvil_grids (np.ndarray): Anvil mask grid data with shape (time, lat, lon).
        met_grids (dict): Dictionary of meteorological variable grids with shape (time, lat, lon).
        times (np.ndarray): Array of datetime64 representing times for each grid.
        interval_type (str): Type of interval ('monthly' or '8day').

    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create latitude and longitude arrays
    latitudes = np.linspace(-90, 90, insitu_grids.shape[1]).astype(
        np.float32
    )
    longitudes = np.linspace(-180, 180, insitu_grids.shape[2]).astype(
        np.float32
    )

    # Create DataArrays for clouds
    insitu_da = xr.DataArray(
        insitu_grids.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    re_insitu_da = xr.DataArray(
        re_insitu_grids.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    IWC_insitu_da = xr.DataArray(
        IWC_insitu_grids.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    anvil_da = xr.DataArray(
        anvil_grids.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    re_anvil_da = xr.DataArray(
        re_anvil_grids.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    IWC_anvil_da = xr.DataArray(
        IWC_anvil_grids.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
        },
    )

    # Create DataArrays for meteorological variables
    met_dataarrays = {}
    for var, grid in met_grids.items():
        da = xr.DataArray(
            grid.astype(np.float32),
            dims=["time", "lat", "lon"],
            coords={
                "time": times,
                "lat": latitudes,
                "lon": longitudes,
            },
            name=var,
        )
        met_dataarrays[var] = da

    # Set encoding for compression
    encoding = {
        "insitu_mask": dict(zlib=True, complevel=4),
        "anvil_mask": dict(zlib=True, complevel=4),
        "re_insitu_mask": dict(zlib=True, complevel=4),
        "re_anvil_mask": dict(zlib=True, complevel=4),
        "IWC_insitu_mask": dict(zlib=True, complevel=4),
        "IWC_anvil_mask": dict(zlib=True, complevel=4),
    }
    # Add encoding for meteorological variables
    for var in met_dataarrays:
        encoding[var] = dict(zlib=True, complevel=4)

    # Create dataset and save to NetCDF
    ds = xr.Dataset(
        {
            "insitu_mask": insitu_da,
            "re_insitu_mask": re_insitu_da,
            "IWC_insitu_mask": IWC_insitu_da,
            "anvil_mask": anvil_da,
            "re_anvil_mask": re_anvil_da,
            "IWC_anvil_mask": IWC_anvil_da,
            **met_dataarrays,
        }
    )

    # Define the output filename based on interval type
    if interval_type == "8day":
        filename = (
            f"{output_dir}/monthly_means_met_data_5degree_test.nc"
        )
    else:
        filename = f"{output_dir}/monthly_weighted_cirrus_class_with_met_data.nc"

    ds.to_netcdf(
        filename, mode="w", engine="h5netcdf", encoding=encoding
    )

    # Save memory
    del ds
    gc.collect()

    # Print message
    print(f"Saved {filename}")


########################################################################################

if __name__ == "__main__":

    pass


########################################################################################
##### End of script ####################################################################
########################################################################################
