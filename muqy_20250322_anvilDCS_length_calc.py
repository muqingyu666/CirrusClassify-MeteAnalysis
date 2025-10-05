"""
# @ Author: Qingyu Mu
# @ Create Time: 2025-03-22 14:51
# @ Modified by: Qingyu Mu
# @ Modified time: 2025-03-22 14:51
# @ Description: In case anything gets wrong, contact me at: muqy20@gmail.com

"""

import gc
import glob
import os
import warnings
from multiprocessing import Pool
from scipy.ndimage import binary_dilation, label

import numpy as np
import xarray as xr
from muqy_20240312_generate_cirrus_class_grid import (
    extract_start_date,
)

# Disable all warnings
warnings.filterwarnings("ignore")

########################################################################################


def calculate_cloud_incidence(cloud_mask, height):
    """
    Calculate the Height-weighted, height-integrated 3D to 2D cloud data projection.
    """
    # Mask for valid heights
    valid_height_mask = (height > 0) & ~np.isnan(height)

    # Use only valid heights and cloud_mask layers
    valid_cloud_mask = cloud_mask[:, valid_height_mask]
    cloud_sum = np.nansum(valid_cloud_mask, axis=1)

    # If at least 1 height bin contains solid cloud (Cld_frac > 0.5), then the cloud is present
    cloud_incidence = np.where(cloud_sum > 30, 1, 0)

    return cloud_incidence.astype(np.int8)


def process_1d_mask(mask, structure=None, iterations=2):
    """
    Perform binary dilation on the input 1D mask, then compute connected
    components, and return an array indicating the length of each connected
    component for every position in that component.

    mask:        1D Boolean numpy array
    structure:   The structuring element used for binary dilation.
                 If None, a default 1D structure will be used.
    iterations:  Number of dilation iterations.

    Return: length_array, labeled_array, num_components
    length_array: 1D array of the same shape as 'mask', each 'True' cluster
                  position holds the size of that connected region
    labeled_array: 1D array of the same shape as 'mask', each cluster assigned a label
    num_components: total number of connected components found
    """
    # If no structure is provided, create a default 1D structuring element
    if structure is None:
        # This is a 5-element structure [True, True, True]
        # that will expand a 'True' by one element on each side per iteration
        structure = np.array([True, True, True, True, True])

    # 1) Binary dilation
    #    This will expand the True regions in the mask
    dilated_mask = binary_dilation(
        mask, structure=structure, iterations=iterations
    )

    # 2) Connected component labeling
    #    'label' returns an array with connected components labeled as 1..num_labels
    #    positions with 0 indicate False or background
    labeled_array, num_components = label(dilated_mask)

    # 3) Create an array to hold the length of each connected component
    length_array = np.zeros_like(labeled_array, dtype=np.int64)

    # 4) For each connected component, find the indices and compute its size
    for comp_label in range(1, num_components + 1):
        # find all positions belonging to this component
        indices = np.where(labeled_array == comp_label)[0]

        # compute the size (length) of this connected component
        region_size = len(indices)

        # fill those positions in length_array with the component size
        length_array[indices] = region_size

    return length_array, labeled_array, num_components


def process_calculation_and_save(input_file_path, output_file_path):
    """
    Process a single NetCDF file: compute weighted cloud fractions and save the modified dataset.
    """
    try:
        print(f"Processing {input_file_path}...")
        #############################################################################
        # Open the input dataset
        with xr.open_dataset(input_file_path, engine="h5netcdf") as ds:
            # Extract necessary variables
            latitude = ds["latitude"].values.astype(np.float32)
            longitude = ds["longitude"].values.astype(np.float32)
            height = ds["height"].values.astype(np.float32)
            insitu_mask = ds["insitu_mask"].values.astype(np.float32)
            anvil_mask = ds["anvil_mask"].values.astype(np.float32)
            DCS_mask = ds["DCS_mask"].values.astype(np.float32)

            # Project profile data to array
            insitu_mask_arr = np.max(insitu_mask, axis=1)
            anvil_mask_arr = np.max(anvil_mask, axis=1)
            DCS_mask_arr = np.max(DCS_mask, axis=1)

            # Replace invalid values with NaN
            height[height < 0] = np.nan

            # Create arr-mask
            insitu_mask_arr = np.where(
                insitu_mask_arr > 20, True, False
            )
            anvil_mask_arr = np.where(anvil_mask_arr > 20, True, False)
            DCS_mask_arr = np.where(DCS_mask_arr > 10, True, False)

            # Combined mask for anvil-DCS
            anvil_DCS_mask_arr = np.logical_or(
                anvil_mask_arr, DCS_mask_arr
            )

            structure_1d = np.array([True, True, True])
            iterations_count = 1

            #############################################################################
            # (A) insitu_mask
            insitu_length_arr, insitu_labels, insitu_num_components = (
                process_1d_mask(
                    insitu_mask_arr,
                    structure=structure_1d,
                    iterations=iterations_count,
                )
            )

            # (B) anvil_mask
            anvil_length_arr, anvil_labels, anvil_num_components = (
                process_1d_mask(
                    anvil_mask_arr,
                    structure=structure_1d,
                    iterations=iterations_count,
                )
            )

            # (C) DCS_mask
            DCS_length_arr, DCS_labels, DCS_num_components = (
                process_1d_mask(
                    DCS_mask_arr,
                    structure=structure_1d,
                    iterations=iterations_count,
                )
            )

            # (D) anvil_DCS_mask
            (
                anvil_DCS_length_arr,
                anvil_DCS_labels,
                anvil_DCS_num_components,
            ) = process_1d_mask(
                anvil_DCS_mask_arr,
                structure=structure_1d,
                iterations=iterations_count,
            )

            #############################################################################
            # Create DataArrays for the new weighted variables
            insitu_mask_da = xr.DataArray(
                insitu_mask_arr.astype(np.int8),
                dims=["time"],
                coords={"time": ds["time"]},
                name="insitu_mask_2D",
                attrs={
                    "description": "2D in-situ cirrus cloud mask",
                },
            )

            insitu_length_da = xr.DataArray(
                insitu_length_arr.astype(np.int32),
                dims=["time"],
                coords={"time": ds["time"]},
                name="insitu_mask_length_2D",
                attrs={
                    "dtype": "int32",
                    "description": "Length of each connected region for 2D in-situ mask",
                },
            )

            anvil_mask_da = xr.DataArray(
                anvil_mask_arr.astype(np.int8),
                dims=["time"],
                coords={"time": ds["time"]},
                name="anvil_mask_2D",
                attrs={
                    "description": "2D anvil cirrus cloud mask",
                },
            )

            anvil_length_da = xr.DataArray(
                anvil_length_arr.astype(np.int32),
                dims=["time"],
                coords={"time": ds["time"]},
                name="anvil_mask_length_2D",
                attrs={
                    "dtype": "int32",
                    "description": "Length of each connected region for 2D anvil mask",
                },
            )

            # (C) DCS
            DCS_mask_da = xr.DataArray(
                DCS_mask_arr.astype(np.int8),
                dims=["time"],
                coords={"time": ds["time"]},
                name="DCS_mask_2D",
                attrs={
                    "description": "2D DCS cloud mask",
                },
            )

            DCS_length_da = xr.DataArray(
                DCS_length_arr.astype(np.int32),
                dims=["time"],
                coords={"time": ds["time"]},
                name="DCS_mask_length_2D",
                attrs={
                    "dtype": "int32",
                    "description": "Length of each connected region for 2D DCS mask",
                },
            )

            # (D) anvil_DCS
            anvil_DCS_mask_da = xr.DataArray(
                anvil_DCS_mask_arr.astype(np.int8),
                dims=["time"],
                coords={"time": ds["time"]},
                name="anvil_DCS_mask_2D",
                attrs={
                    "description": "2D combined anvil+DCS mask",
                },
            )

            anvil_DCS_length_da = xr.DataArray(
                anvil_DCS_length_arr.astype(np.int32),
                dims=["time"],
                coords={"time": ds["time"]},
                name="anvil_DCS_mask_length_2D",
                attrs={
                    "dtype": "int32",
                    "description": "Length of each connected region for 2D anvil+DCS mask",
                },
            )

            # Store the latitude and longitude as DataArrays
            latitude_da = xr.DataArray(
                latitude.astype(np.float32),
                dims=["time"],
                coords={"time": ds["time"]},
                name="latitude",
            )

            longitude_da = xr.DataArray(
                longitude.astype(np.float32),
                dims=["time"],
                coords={"time": ds["time"]},
                name="longitude",
            )

            # Create a new dataset with the modified variables
            ds_new = xr.Dataset(
                {
                    "insitu_mask_2D": insitu_mask_da,
                    "insitu_mask_length_2D": insitu_length_da,
                    "anvil_mask_2D": anvil_mask_da,
                    "anvil_mask_length_2D": anvil_length_da,
                    "DCS_mask_2D": DCS_mask_da,
                    "DCS_mask_length_2D": DCS_length_da,
                    "anvil_DCS_mask_2D": anvil_DCS_mask_da,
                    "anvil_DCS_mask_length_2D": anvil_DCS_length_da,
                    "latitude": latitude_da,
                    "longitude": longitude_da,
                }
            )

            # Global attributes
            ds_new.attrs["description"] = (
                "New dataset containing masks and their connected region lengths"
            )
            ds_new.attrs["author"] = "Qingyu Mu"
            ds_new.attrs["creation_time"] = str(np.datetime64("now"))

            #############################################################################
            # Define the output file path
            output_file = os.path.join(
                output_file_path,
                os.path.basename(input_file_path).replace(
                    "processed", "Maks_length"
                ),
            )

            # Define compression options
            comp = dict(zlib=True, complevel=2)

            # Apply compression to all data variables
            encoding = {var: comp for var in ds_new.data_vars}

            # Save the modified dataset to a new NetCDF file
            ds_new.to_netcdf(
                output_file,
                mode="w",
                engine="h5netcdf",
                encoding=encoding,
            )

            print(f"Saved {output_file}")

        # Cleanup through garbage collection
        del (
            ds,
            ds_new,
            insitu_mask,
            anvil_mask,
            DCS_mask,
            insitu_mask_arr,
            anvil_mask_arr,
            DCS_mask_arr,
            insitu_length_arr,
            insitu_labels,
            insitu_num_components,
            anvil_length_arr,
            anvil_labels,
            anvil_num_components,
            DCS_length_arr,
            DCS_labels,
            DCS_num_components,
            anvil_DCS_length_arr,
            anvil_DCS_labels,
            anvil_DCS_num_components,
            insitu_mask_da,
            insitu_length_da,
            anvil_mask_da,
            anvil_length_da,
            DCS_mask_da,
            DCS_length_da,
            anvil_DCS_mask_da,
            anvil_DCS_length_da,
        )
        # Collect garbage to free memory
        gc.collect()

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")


def main_execution(input_file_paths, output_file_path, num_workers=4):
    """
    Main function to process multiple NetCDF files in parallel.
    """
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # Ensure output directory exists
    os.makedirs(output_file_path, exist_ok=True)

    # with Pool(processes=num_workers) as pool:
    #     pool.starmap(
    #         process_calculation_and_save,
    #         [
    #             (input_file_path, output_file_path)
    #             for input_file_path in input_file_paths
    #         ],
    #     )

    for input_file_path in input_file_paths:
        process_calculation_and_save(input_file_path, output_file_path)


########################################################################################


if __name__ == "__main__":
    pass

########################################################################################
