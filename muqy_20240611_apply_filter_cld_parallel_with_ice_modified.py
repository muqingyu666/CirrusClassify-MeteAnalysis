# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:44:22

import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import xarray as xr
from muqy_20240104_filter_anvil_insitu_cirrus import (
    CloudSatProcessor_Ver2,
)


########################################################################################


def get_all_files_by_directory(base_paths):
    """
    Recursively get all files in the given base paths, grouped by their parent directory.

    Args:
        base_paths (dict): A dictionary containing base paths as keys and corresponding paths as values.

    Returns:
        dict: A nested dictionary where the outer keys are day-of-year directories and
            inner keys are dataset names with values being lists of file paths.
    """
    all_file_paths = {}

    for dataset, base_path in base_paths.items():
        for root, _, files in os.walk(base_path):
            if files:
                dir_name = os.path.relpath(root, base_path)
                if dir_name not in all_file_paths:
                    all_file_paths[dir_name] = {}
                if dataset not in all_file_paths[dir_name]:
                    all_file_paths[dir_name][dataset] = []
                all_file_paths[dir_name][dataset].extend(
                    [os.path.join(root, file) for file in files]
                )

    return all_file_paths


def find_common_dirs(
    all_file_paths,
    required_datasets={"CLDCLASS", "ICE", "GEOPROF", "ECMWF_AUX"},
):
    """
    Find the common files among multiple sets of file paths, keeping only those with matching time parts.
    Ensure all datasets have corresponding files before keeping the current directory and files.

    Args:
    all_file_paths (dict): A dictionary containing sets of file paths.

    Returns:
    dict: A dictionary containing the common files for each set of file paths.
    """
    required_datasets = required_datasets
    common_files = {}

    for day_dir, datasets in all_file_paths.items():
        if not required_datasets.issubset(datasets.keys()):
            continue  # Skip if not all required datasets are present

        # Extract the time parts of the file names for each dataset
        file_identifiers = {
            dataset: set(os.path.basename(f)[:13] for f in paths)
            for dataset, paths in datasets.items()
        }

        # Find the intersection of these time parts across all datasets
        common_identifiers = set.intersection(*file_identifiers.values())

        if common_identifiers:
            common_files[day_dir] = {
                dataset: sorted(
                    [
                        f
                        for f in paths
                        if os.path.basename(f)[:13] in common_identifiers
                    ],
                    key=lambda x: os.path.basename(x)[:13],
                )
                for dataset, paths in datasets.items()
            }

    # Remove directories with no common files
    to_remove = [
        dir_name
        for dir_name, datasets in all_file_paths.items()
        if dir_name not in common_files
    ]
    for dir_name in to_remove:
        del all_file_paths[dir_name]

    return common_files


def process_directory(processing_path):
    """
    Process files in a directory and return the results.

    Args:
        common_files (dict): Common files grouped by directory.
        file_path (str): The path to save processed data.
        day_dir (str): The directory to process.
        initial_offset (int): Initial time offset.

    Returns:
        list: Processed results.
        int: Updated offset.
    """

    # Initialize time offset
    dir_results = []

    # Get the number of files in the directory
    file_num = len(processing_path["CLDCLASS"])

    # Process each dataset in the directory
    # That is, each date in the directory
    # Within each dates are multiple files, like 2006/164 contains multiple files
    for file_idx in range(file_num):

        # Process the CloudSat data and extract necessary datasets
        result = cloudsat_data_processing(processing_path, file_idx)

        # Print the file index
        print(f"File {file_idx + 1}/{file_num} processed")

        # Append the result to the list
        dir_results.append(result)

    return dir_results


def save_and_cleanup(dir_results, file_path, dir):
    """
    Save the batch results and perform cleanup.

    Args:
        dir_results (list): List of batch results.
        file_path (str): Path to the file.
        dir (str): Directory.

    Returns:
        None
    """
    # Save the batch results if not empty
    if dir_results:
        year, day_of_year = dir.split("/")
        date = pd.to_datetime(f"{year}-{day_of_year}", format="%Y-%j").strftime(
            "%Y_%m_%d"
        )
        file_name = f"{file_path}/processed_cloudsat_data_{date}.nc"
        concatenate_and_save_batch_results(
            dir_results,
            dir_results[0][17],
            dir_results[0][18],
            file_name,
        )
        print(f"Batch saved for {date}")

        # Perform cleanup
        del dir_results, file_name, date


def process_and_save(common_paths, file_path, dir):
    """
    Process the files in a directory and save the results.

    Args:
        common_paths (dict): Common files grouped by directory.
        file_path (str): The path to save processed data.
        dir (str): The directory to process.

    Returns:
        None
    """
    # Get the processing path
    processing_path = common_paths[dir]

    # Process the directory
    dir_results = process_directory(processing_path)

    # Save the results and perform cleanup
    save_and_cleanup(dir_results, file_path, dir)


def batchProcessFiles_parallel(
    common_paths,
    file_path="/RAID01/data/CloudSat_Cirrus_classification",
    num_processes_level1=40,
):
    """
    Process a batch of files in parallel, grouped by directories.

    Args:
        common_files (dict): Common files grouped by directory.
        file_path (str): The path to save processed data.
        num_processes (int): Number of parallel processes.

    Returns:
        None
    """
    # Create a directory to save the processed data
    os.makedirs(file_path, exist_ok=True)

    # Get the list of directories, the date is in the format "YYYY/DOY"
    # e.g., "2010/001" for the first day of 2010
    # 2006/164, 2006/165 ...
    directories = list(common_paths.keys())

    # Process directories in parallel
    with mp.Pool(processes=min(num_processes_level1, len(directories))) as pool:
        pool.starmap(
            process_and_save,
            [(common_paths, file_path, dir) for dir in directories],
        )


def cloudsat_data_processing(common_files, file_idx):
    """
    Process CloudSat data and extract necessary information.

    Args:
        common_files (list): List of common files.
        file_idx (int): Index of the file to process.
        last_time_offset (float): Last time offset.

    Returns:
        tuple: A tuple containing the following elements:
            - time (ndarray): Array of time values.
            - elv (ndarray): Array of elevation values.
            - lat (ndarray): Array of latitude values.
            - lon (ndarray): Array of longitude values.
            - hgt (ndarray): Array of height values.
            - insitu_mask (ndarray): Array representing the insitu mask.
            - anvil_mask (ndarray): Array representing the anvil mask.
            - next_time_offset (float): Next time offset.
            - satellite_start_time (str or None): Start time of the satellite data.
            - satellite_end_time (str or None): End time of the satellite data.
    """
    # Initialize CloudSatProcessor and process data
    processor = CloudSatProcessor_Ver2(
        common_files,
        structure_0=np.ones((11, 1)),  # horizontal extent
        structure_1=np.ones((1, 4)),  # vertical extent
    )

    # Process the CloudSat data
    data, cirrus_insitu_mask, cirrus_anvil_mask, DCS_mask = (
        processor.main_process(
            file_index=file_idx,
            dataset_names=["CLDCLASS", "ICE", "GEOPROF", "ECMWF_AUX"],
            iterations=25,
        )
    )

    # Extract necessary data and apply time offset
    batch_start_time = pd.to_datetime(
        data.get("start_time"), format="%Y_%m_%d_%H_%M"
    )

    # Create a new datetime array based on the time dimension
    # Convert time from seconds to datetime
    new_time = batch_start_time + pd.to_timedelta(data["time"], unit="s")

    # Generate insitu and anvil masks based on the processor's output
    cirrus_insitu_mask = np.where(
        cirrus_insitu_mask == 1, data["cld_frac"], cirrus_insitu_mask
    ).astype(np.int8)
    cirrus_anvil_mask = np.where(
        cirrus_anvil_mask == 1, data["cld_frac"], cirrus_anvil_mask
    ).astype(np.int8)
    DCS_mask = np.where(DCS_mask, data["cld_frac"], DCS_mask).astype(np.int8)

    return (
        new_time,
        data["elv"],
        data["lat"],
        data["lon"],
        data["hgt"],
        data["skin_temperature"],
        cirrus_insitu_mask,
        cirrus_anvil_mask,
        DCS_mask,
        data["re"],
        data["IWC"],
        data["EXT_coef"],
        data["temperature"],
        data["specific_humidity"],
        data["u_velocity"],
        data["v_velocity"],
        data["pressure"],
        # Capture satellite start and end times
        data.get("start_time"),
        data.get("end_time"),
    )


def concatenate_and_save_batch_results(
    batch_results,
    batch_start_time,
    batch_end_time,
    file_name="/RAID01/data/CloudSat_Cirrus_classification/processed_cloudsat_data.nc",
):
    # Check if batch_results is empty to prevent errors
    if not batch_results:
        print("No results to save.")
        return

    # Print the number of files in the batch
    print(f"Concatenate batch with {len(batch_results)} files.")

    # Unpack the batch results
    (
        times,
        elvs,
        lats,
        lons,
        heights,
        skin_temperature,
        insitu_masks,
        anvil_masks,
        DCS_mask,
        re,
        IWC,
        EXT_coef,
        temperature,
        specific_humidity,
        u_velocity,
        v_velocity,
        pressure,
        _,
        _,
    ) = zip(*batch_results)

    # Convert the lists to numpy arrays
    times = np.concatenate(times)
    elvs = np.concatenate(elvs).astype(np.float16)
    lats = np.concatenate(lats).astype(np.float32)
    lons = np.concatenate(lons).astype(np.float32)
    skin_temperature = np.concatenate(skin_temperature).astype(np.float32)
    insitu_masks = np.concatenate(insitu_masks).astype(np.int8)
    anvil_masks = np.concatenate(anvil_masks).astype(np.int8)
    DCS_masks = np.concatenate(DCS_mask).astype(np.int8)
    re = np.concatenate(re).astype(np.float32)
    IWC = np.concatenate(IWC).astype(np.float32)
    EXT_coef = np.concatenate(EXT_coef).astype(np.float32)
    temperatures = np.concatenate(temperature).astype(np.float32)
    specific_humiditys = np.concatenate(specific_humidity).astype(np.float32)
    u_winds = np.concatenate(u_velocity).astype(np.float32)
    v_winds = np.concatenate(v_velocity).astype(np.float32)
    pressures = np.concatenate(pressure).astype(np.float16)

    # Create xarray Dataset from concatenated arrays
    ds = xr.Dataset(
        {
            "elevation": (["time"], elvs),
            "latitude": (["time"], lats, {"dtype": "float32"}),
            "longitude": (["time"], lons, {"dtype": "float32"}),
            "insitu_mask": (
                ["time", "height"],
                insitu_masks,
                {"dtype": "int8"},
            ),
            "anvil_mask": (
                ["time", "height"],
                anvil_masks,
                {"dtype": "int8"},
            ),
            "DCS_mask": (
                ["time", "height"],
                DCS_masks,
                {"dtype": "int8"},
            ),
            "re": (["time", "height"], re, {"dtype": "float32"}),
            "IWC": (["time", "height"], IWC, {"dtype": "float32"}),
            "EXT_coef": (
                ["time", "height"],
                EXT_coef,
                {"dtype": "float32"},
            ),
            "skin_temperature": (
                ["time"],
                skin_temperature,
                {"dtype": "float32"},
            ),
            "temperature": (
                ["time", "height"],
                temperatures,
                {"dtype": "float32"},
            ),
            "specific_humidity": (
                ["time", "height"],
                specific_humiditys,
                {"dtype": "float32"},
            ),
            "u_wind": (
                ["time", "height"],
                u_winds,
                {"dtype": "float32"},
            ),
            "v_wind": (
                ["time", "height"],
                v_winds,
                {"dtype": "float32"},
            ),
            "pressure": (
                ["time", "height"],
                pressures,
                {"dtype": "float32"},
            ),
        },
        coords={
            "time": times,
            # height is constant for all files
            "height": heights[0].astype(np.float32),
        },
    )

    # Set metadata and attributes
    ds.attrs["batch_start_time"] = pd.to_datetime(batch_start_time).strftime(
        "%Y_%m_%d_%H_%M"
    )
    ds.attrs["batch_end_time"] = pd.to_datetime(batch_end_time).strftime(
        "%Y_%m_%d_%H_%M"
    )

    # Define compression options
    comp = dict(zlib=True, complevel=4)  # Adjust complevel as needed

    # Apply compression to all variables
    encoding = {var: comp for var in ds.data_vars}

    # Save the dataset to a NetCDF file
    ds.to_netcdf(file_name, engine="h5netcdf", encoding=encoding, mode="w")
    print(f"Data saved to {file_name}")


########################################################################################


if __name__ == "__main__":

    # ----------------------------------------------------------------
    # Set file paths and other parameters
    # ----------------------------------------------------------------

    pass


########################################################################################
