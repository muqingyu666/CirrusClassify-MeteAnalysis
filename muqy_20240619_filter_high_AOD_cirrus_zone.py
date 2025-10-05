# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:44:30


import glob
import os
import warnings
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from muqy_20240312_generate_cirrus_class_grid import (
    extract_start_date,
)

# Disable all warnings
warnings.filterwarnings("ignore")

########################################################################################
##### Functions for filter high AOD cirrus zone ########################################
########################################################################################


def process_time_point(
    aerosol_data, aerosol_values, cirrus_data, time_points, threshold
):
    """
    Process a chunk of time points in the cirrus data in a vectorized manner.

    Parameters:
    - aerosol_data (xarray.Dataset): The aerosol data.
    - cirrus_data_chunk (xarray.Dataset): The chunk of cirrus data.
    - time_indices_chunk (list): List of time indices in the chunk.
    - threshold (float): Threshold value for filtering cirrus data based on aerosol data.

    Returns:
    None
    """
    cirrus_lon = cirrus_data["longitude"].values
    cirrus_lat = cirrus_data["latitude"].values

    # Get the corresponding aerosol data time indices
    aerosol_time_indices = np.searchsorted(
        aerosol_data["time"].values, time_points
    )
    aerosol_time_indices = np.clip(
        aerosol_time_indices, 0, len(aerosol_data["time"].values) - 1
    )

    # Get the corresponding grid points in aerosol data
    aerosol_lon_indices = np.searchsorted(
        aerosol_data["lon"].values, cirrus_lon
    )
    aerosol_lon_indices = np.clip(
        aerosol_lon_indices, 0, len(aerosol_data["lon"].values) - 1
    )
    aerosol_lat_indices = np.searchsorted(
        aerosol_data["lat"].values, cirrus_lat
    )
    aerosol_lat_indices = np.clip(
        aerosol_lat_indices, 0, len(aerosol_data["lat"].values) - 1
    )

    # Extract the relevant aerosol data
    selected_aerosol_values = aerosol_values[
        aerosol_time_indices, aerosol_lat_indices, aerosol_lon_indices
    ]

    # Find the indices where the condition is met
    # AKA where the aerosol values are greater than the threshold
    condition_met_indices = np.where(
        selected_aerosol_values > threshold
    )[0]

    # Apply the mask and set other variables to NaN
    cirrus_data["insitu_mask"].values[condition_met_indices] = -1
    cirrus_data["anvil_mask"].values[condition_met_indices] = -1

    # Apply the mask to all other variables
    for var in cirrus_data.data_vars:
        if var in [
            "re",
            "IWC",
            "EXT_coef",
        ]:
            cirrus_data[var].values[condition_met_indices] = np.nan


def filter_cirrus_data(
    aerosol_data,
    cirrus_file,
    output_folder,
    threshold=0.1,
):
    """
    Filter cirrus data based on aerosol data.

    Parameters:
    - aerosol_data (xarray.Dataset): The aerosol data.
    - cirrus_file (str): Path to the cirrus data file.
    - output_folder (str): Path to the folder where filtered data should be saved.
    - threshold (float): Threshold value for filtering cirrus data based on aerosol data.
    - internal_workers (int): Number of parallel workers for internal loop.

    Returns:
    None
    """
    cirrus_data = xr.open_dataset(cirrus_file, engine="h5netcdf")
    time_points = cirrus_data["time"].values
    aerosol_values = aerosol_data["TOTEXTTAU"].values

    # Prepare arguments for parallel processing
    args = (
        aerosol_data,
        aerosol_values,
        cirrus_data,
        time_points,
        threshold,
    )

    # Parallelize the processing of each time point using multiprocessing.Pool
    process_time_point(*args)

    # Save the modified cirrus data to the new directory
    output_file = os.path.join(
        output_folder, os.path.basename(cirrus_file)
    )

    # Define compression options
    comp = dict(zlib=True, complevel=4)  # Adjust complevel as needed

    # Apply compression to all variables
    encoding = {var: comp for var in cirrus_data.data_vars}

    cirrus_data.to_netcdf(
        output_file, mode="w", engine="h5netcdf", encoding=encoding
    )
    print(f"Saved filtered data to {output_file}")


def process_cirrus_file(
    aerosol_file,
    cirrus_file,
    output_folder,
    threshold,
):
    aerosol_data = xr.open_dataset(aerosol_file, engine="h5netcdf")
    filter_cirrus_data(
        aerosol_data,
        cirrus_file,
        output_folder,
        threshold,
    )


def process_aerosol_cirrus(
    aerosol_file,
    cirrus_files,
    cirrus_dates,
    output_folder,
    threshold=0.1,
    external_workers=4,
):
    """
    Process aerosol and cirrus data by filtering cirrus data based on aerosol data in parallel.

    Parameters:
    - aerosol_file (str): Path to the aerosol data file.
    - cirrus_files (list): List of paths to the cirrus data files.
    - cirrus_dates (list): List of dates extracted from cirrus file names.
    - output_folder (str): Path to the folder where filtered data should be saved.
    - threshold (float): Threshold value for filtering cirrus data based on aerosol data.
    - external_workers (int): Number of parallel workers for external loop.
    - internal_workers (int): Number of parallel workers for internal loop.

    Returns:
    None
    """
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Load aerosol data to get the time range
    aerosol_data = xr.open_dataset(aerosol_file, engine="h5netcdf")
    start_date = np.datetime64(aerosol_data["time"].values[0], "D")
    end_date = np.datetime64(aerosol_data["time"].values[-1], "D")

    # Filter cirrus files within the date range of the aerosol data
    filtered_cirrus_files = [
        cirrus_files[i]
        for i, date in enumerate(cirrus_dates)
        if start_date <= date <= end_date
    ]

    # Check if there are any filtered cirrus files
    if not filtered_cirrus_files:
        print(
            f"No cirrus data within the time range of aerosol file: {aerosol_file}"
        )
        return

    # Process each filtered cirrus file in parallel
    with ProcessPoolExecutor(max_workers=external_workers) as executor:
        futures = [
            executor.submit(
                process_cirrus_file,
                aerosol_file,
                cf,
                output_folder,
                threshold,
            )
            for cf in filtered_cirrus_files
        ]
        for future in futures:
            future.result()


########################################################################################

if __name__ == "__main__":
    pass
