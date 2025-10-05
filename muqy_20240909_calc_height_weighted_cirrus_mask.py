# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-08-29 08:49
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:44:52

import gc
import glob
import os
import warnings
from multiprocessing import Pool

import numpy as np
import xarray as xr
from muqy_20240312_generate_cirrus_class_grid import (
    extract_start_date,
)

# Disable all warnings
warnings.filterwarnings("ignore")


########################################################################################


def calculate_weighted_cloud_amount(cloud_mask, height):
    """
    Calculate the Height-weighted, height-integrated 3D to 2D cloud data projection.
    """
    # Mask for valid heights
    valid_height_mask = (height > 0) & ~np.isnan(height)

    # Use only valid heights and cloud_mask layers
    valid_height = height[valid_height_mask]
    valid_cloud_mask = cloud_mask[:, valid_height_mask]

    # Compute layer thicknesses for valid heights
    layer_thickness = np.diff(valid_height)

    # Calculate the cloud fraction for each profile
    cloud_fraction_2d = (
        valid_cloud_mask * layer_thickness / np.sum(layer_thickness)
    )

    return cloud_fraction_2d.astype(np.float32)


def calculate_weighted_cloud_amount(cloud_mask, height):
    """
    Calculate the Height-weighted, height-integrated 3D to 2D cloud data projection.
    """
    # Mask for valid heights
    valid_height_mask = (height > 0) & ~np.isnan(height)

    # Use only valid heights and cloud_mask layers
    valid_height = height[valid_height_mask]
    valid_cloud_mask = cloud_mask[:, valid_height_mask]

    # Compute layer thicknesses for valid heights
    layer_thickness = np.diff(valid_height)

    # Avoid division by zero
    total_thickness = np.nansum(layer_thickness)

    if total_thickness == 0:
        return np.full(
            cloud_mask.shape[0], np.nan, dtype=np.float32
        )

    # Calculate the cloud fraction for each profile
    cloud_fraction_2d = (
        valid_cloud_mask[:, :-1] * layer_thickness
    ) / total_thickness

    # Sum over the height dimension to get 2D fraction
    cloud_fraction_2d = np.nansum(cloud_fraction_2d, axis=1)

    return cloud_fraction_2d.astype(np.float32)


def process_calculation_and_save(input_file_path, output_file_path):
    """
    Process a single NetCDF file: compute weighted cloud fractions and save the modified dataset.
    """
    try:
        # Open the input dataset
        with xr.open_dataset(
            input_file_path, engine="h5netcdf"
        ) as ds:
            # Extract necessary variables
            height = ds["height"].values.astype(np.float32)
            insitu_mask = ds["insitu_mask"].values.astype(
                np.float32
            )
            anvil_mask = ds["anvil_mask"].values.astype(np.float32)

            # Replace invalid values with NaN
            insitu_mask[insitu_mask <= -1] = np.nan
            anvil_mask[anvil_mask <= -1] = np.nan
            height[height < 0] = np.nan

            # Calculate weighted cloud amounts
            insitu_weighted = calculate_weighted_cloud_amount(
                insitu_mask, height
            )
            anvil_weighted = calculate_weighted_cloud_amount(
                anvil_mask, height
            )

            # Create DataArrays for the new weighted variables
            insitu_weighted_da = xr.DataArray(
                insitu_weighted,
                dims=["time"],
                coords={"time": ds["time"]},
                name="insitu_fraction_weighted_2D",
                attrs={
                    "dtype": "float32",
                    "description": "Height-weighted 2D insitu cirrus cloud occurrence",
                },
            )

            anvil_weighted_da = xr.DataArray(
                anvil_weighted,
                dims=["time"],
                coords={"time": ds["time"]},
                name="anvil_fraction_weighted_2D",
                attrs={
                    "dtype": "float32",
                    "description": "Height-weighted 2D anvil cirrus cloud occurrence",
                },
            )

            # Create a copy of the dataset to modify
            ds_modified = ds.copy()

            # Replace the original mask variables with the weighted ones
            ds_modified = ds_modified.drop_vars(
                ["insitu_mask", "anvil_mask"]
            )
            ds_modified = ds_modified.assign(
                insitu_fraction_weighted_2D=insitu_weighted_da,
                anvil_fraction_weighted_2D=anvil_weighted_da,
            )

            # Define the output file path
            output_file = os.path.join(
                output_file_path,
                os.path.basename(input_file_path).replace(
                    "processed", "weighted"
                ),
            )

            # Define compression options
            comp = dict(zlib=True, complevel=4)

            # Apply compression to all data variables
            encoding = {var: comp for var in ds_modified.data_vars}

            # Save the modified dataset to a new NetCDF file
            ds_modified.to_netcdf(
                output_file,
                mode="w",
                engine="h5netcdf",
                encoding=encoding,
            )

            print(f"Saved {output_file}")

        # Cleanup
        del (
            ds,
            ds_modified,
            insitu_mask,
            anvil_mask,
            height,
            insitu_weighted,
            anvil_weighted,
        )

        gc.collect()

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")


def main_execution(
    input_file_paths, output_file_path, num_workers=4
):
    """
    Main function to process multiple NetCDF files in parallel.
    """
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # Ensure output directory exists
    os.makedirs(output_file_path, exist_ok=True)

    # Process files in parallel
    print("Processing files...")

    with Pool(processes=num_workers) as pool:
        pool.starmap(
            process_calculation_and_save,
            [
                (input_file_path, output_file_path)
                for input_file_path in input_file_paths
            ],
        )


########################################################################################


if __name__ == "__main__":
    pass

########################################################################################
