# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-07-09 16:34
# @Last Modified by:   Muqy
# @Last Modified time: 2024-09-22 15:08


import gc
import glob
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
from EC_AUX_deciles_dict import intervals_dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Disable all warnings
warnings.filterwarnings("ignore")

########################################################################################

# Constants
DATA_DIR = "/RAID01/data/PROJECT_CIRRUS_CLASSIFICATION/Calced_Vars_Cirrus_classification_233K_1500m_overlay_aerosol_0085"
OUTPUT_DIR_PROFILE = "/RAID01/data/PROJECT_CIRRUS_CLASSIFICATION/Calced_ECAUX_classification_cirrus_profile_233K_1500m_overlay_aerosol_0085"
OUTPUT_DIR_INCIDENCE = "/RAID01/data/PROJECT_CIRRUS_CLASSIFICATION/Calced_ECAUX_classification_cirrus_incidence_233K_1500m_overlay_aerosol_0085"

# DATA_DIR = "E:/Data/Calced_Vars_Cirrus_classification_233K_1920m_overlay_aerosol_0085"
# OUTPUT_DIR_PROFILE = "E:/Data/Calced_ECAUX_classification_cirrus_profile_233K_1920m_overlay_aerosol_0085"
# OUTPUT_DIR_INCIDENCE = "E:/Data/Calced_ECAUX_classification_cirrus_incidence_233K_1920m_overlay_aerosol_0085"

# Alternatively, uncomment and adjust the following paths for different environments
# DATA_DIR = "../Data_python/CloudSat_data/Calced_Vars_Cirrus_classification_233K_1440m_overlay_aerosol_008"
# OUTPUT_DIR_PROFILE = "../Data_python/CloudSat_data/Calced_ECAUX_classification_cirrus_profile_overlay_aerosol_008"
# OUTPUT_DIR_INCIDENCE = "../Data_python/CloudSat_data/Calced_ECAUX_classification_cirrus_incidence_overlay_aerosol_008"

NUM_WORKERS = 2  # Adjust based on your system's capabilities

########################################################################################


def preprocess_deciles_dict(deciles_dict):
    """Remove the last edge from each variable's range to prepare for binning."""
    return {var: values[:-1] for var, values in deciles_dict.items()}


def generate_bin_labels(ranges):
    """Generate categorical labels for binning based on the provided ranges."""
    bins = pd.cut(
        ranges,
        bins=ranges,
        right=True,
        include_lowest=True,
        precision=5,
    )
    return bins.categories


def create_masks(ds, mask_vars):
    """
    Create processed masks from the dataset.

    Parameters:
        ds (xarray.Dataset): The opened dataset.
        mask_vars (list): List of mask variable names.

    Returns:
        dict: Processed masks.
    """
    masks = {}
    for mask_var in mask_vars:
        mask = ds[mask_var].values.astype(np.float32)
        mask = np.where(mask < 0, np.nan, mask)
        mask = np.where(mask > 40, 1, 0)

        # Create a profile mask by summing along the height axis
        profile_mask = np.nansum(mask, axis=1)

        # For profile results: retain height dimension, filter out profiles with sum < 2
        masks[f"{mask_var}_profile"] = np.where(
            profile_mask[:, np.newaxis] >= 2, mask, 0
        )

        # For incidence results: binary mask per profile
        masks[f"{mask_var}_incidence"] = (profile_mask >= 2).astype(
            np.float32
        )

    return masks


def process_file(
    file_path,
    variable,
    bin_edges,
    bin_labels,
    max_samples_profile=50000,
    max_samples_incidence=10000,
):
    """
    Process a single file to extract both profile and incidence data.

    Parameters:
        file_path (str): Path to the NetCDF file.
        variable (str): Variable name to process.
        bin_edges (np.ndarray): Edges for binning.
        bin_labels (pd.CategoricalIndex): Labels for the bins.
        max_samples_profile (int): Max samples per bin for profile.
        max_samples_incidence (int): Max samples per bin for incidence.

    Returns:
        tuple: (profile_results, incidence_results, height) or (None, None, None) on failure.
    """
    try:
        with xr.open_dataset(file_path, engine="h5netcdf") as ds:
            var_data = ds[variable].values.astype(np.float32)

            # Check if var_data is empty
            if var_data.size == 0:
                logging.warning(
                    f"Skipping file {file_path} for variable {variable}. Data is empty."
                )
                return None, None, None

            # Handle NaN and inf values
            nan_percentage = np.isnan(var_data).mean()
            inf_percentage = np.isinf(var_data).mean()

            if nan_percentage > 0.999 or inf_percentage > 0.999:
                logging.warning(
                    f"Skipping file {file_path} for variable {variable}. "
                    f"{nan_percentage:.3%} of values are NaN (threshold: 99.9%), {inf_percentage:.3%} of values are infinite (threshold: 99.9%)."
                )
                return None, None, None

            results_profile = {str(label): {} for label in bin_labels}
            results_incidence = {str(label): {} for label in bin_labels}
            height = ds["height"].values

            mask_vars = ["insitu_mask", "anvil_mask"]
            masks = create_masks(ds, mask_vars)

            # Iterate over the bin edges and calculate the required statistics
            for i, label in enumerate(bin_labels):
                try:
                    # Define bin range
                    bin_min = bin_edges[i]
                    bin_max = bin_edges[i + 1]

                    # Create mask for the current bin
                    bin_mask = (
                        (var_data >= bin_min)
                        & (var_data < bin_max)
                        & (~np.isnan(var_data))
                    )

                    valid_indices = np.where(bin_mask)[0]

                    # Process Profile Results
                    if valid_indices.size > 0:
                        # Sample indices for profile
                        if valid_indices.size > max_samples_profile:
                            sampled_indices_profile = np.random.choice(
                                valid_indices,
                                size=max_samples_profile,
                                replace=False,
                            )
                        else:
                            sampled_indices_profile = valid_indices

                        # Aggregate profile data
                        for key in mask_vars + [
                            "re_insitu",
                            "re_anvil",
                            "IWC_insitu",
                            "IWC_anvil",
                        ]:
                            if key in mask_vars:
                                profile_key = key
                                data_mean = np.nanmean(
                                    masks[f"{key}_profile"][
                                        sampled_indices_profile
                                    ],
                                    axis=0,
                                )
                            else:
                                profile_key = key
                                data_mean = np.nanmean(
                                    ds[key].values[
                                        sampled_indices_profile
                                    ],
                                    axis=0,
                                )
                            results_profile[str(label)][
                                profile_key
                            ] = data_mean

                    else:
                        # Assign NaNs if no valid data
                        for key in mask_vars + [
                            "re_insitu",
                            "re_anvil",
                            "IWC_insitu",
                            "IWC_anvil",
                        ]:
                            results_profile[str(label)][key] = np.full(
                                height.size,
                                np.nan,
                                dtype=np.float32,
                            )

                    # Process Incidence Results
                    if valid_indices.size > 0:
                        # Sample indices for incidence
                        if valid_indices.size > max_samples_incidence:
                            sampled_indices_incidence = (
                                np.random.choice(
                                    valid_indices,
                                    size=max_samples_incidence,
                                    replace=False,
                                )
                            )
                        else:
                            sampled_indices_incidence = valid_indices

                        # Aggregate incidence data
                        for key in mask_vars:
                            incidence_key = f"{key}_mean"
                            data_mean = np.nanmean(
                                masks[f"{key}_incidence"][
                                    sampled_indices_incidence
                                ],
                                axis=0,
                            )
                            results_incidence[str(label)][
                                incidence_key
                            ] = data_mean
                    else:
                        # Assign NaNs if no valid data
                        for key in mask_vars:
                            incidence_key = f"{key}_mean"
                            results_incidence[str(label)][
                                incidence_key
                            ] = np.nan

                except Exception as e:
                    logging.error(
                        f"Error processing bin {label} for variable {variable} in file {file_path}: {e}"
                    )
                    continue

            del var_data, ds
            gc.collect()

            return results_profile, results_incidence, height

    except KeyError as e:
        logging.error(
            f"KeyError in file {file_path} for variable {variable}: {e}"
        )
        return None, None, None
    except ValueError as e:
        logging.error(
            f"ValueError in file {file_path} for variable {variable}: {e}"
        )
        return None, None, None
    except Exception as e:
        logging.error(
            f"Unexpected error processing file {file_path} for variable {variable}: {e}"
        )
        return None, None, None


def save_results_to_netcdf_profile(
    results, output_dir, variable, height
):
    """Save profile results to NetCDF."""
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        data_vars = {}

        for key in [
            "insitu_mask",
            "anvil_mask",
            "re_insitu",
            "re_anvil",
            "IWC_insitu",
            "IWC_anvil",
        ]:
            if key in results[list(results.keys())[0]]:
                data_vars[key] = (
                    ["range", "height"],
                    np.array(
                        [
                            results[label][key]
                            for label in results.keys()
                        ]
                    ).astype(np.float32),
                )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "range": list(results.keys()),
                "height": height,
            },
        )

        filename = os.path.join(output_dir, f"{variable}.nc")
        ds.to_netcdf(
            filename,
            engine="h5netcdf",
            mode="w",
            encoding={
                var: (
                    {
                        "dtype": "float32",
                        "zlib": True,
                        "complevel": 3,
                    }
                )
                for var in ds.data_vars
            },
        )
        logging.info(f"Saved profile results to {filename}")

    except Exception as e:
        logging.error(
            f"Error saving profile results to NetCDF for {variable}: {e}"
        )

    finally:
        del ds
        gc.collect()


def save_results_to_netcdf_incidence(results, output_dir, variable):
    """Save incidence results to NetCDF."""
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        data_vars = {}

        for key in [
            "insitu_mask_mean",
            "anvil_mask_mean",
        ]:
            if key in results[list(results.keys())[0]]:
                data_vars[key] = (
                    ["range"],
                    np.array(
                        [
                            results[label][key]
                            for label in results.keys()
                        ]
                    ).astype(np.float32),
                )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "range": list(results.keys()),
            },
        )

        filename = os.path.join(
            output_dir, f"{variable}_cirrus_incidence.nc"
        )
        ds.to_netcdf(
            filename,
            engine="h5netcdf",
            mode="w",
            encoding={
                var: (
                    {
                        "dtype": "float32",
                        "zlib": True,
                        "complevel": 3,
                    }
                )
                for var in ds.data_vars
            },
        )
        logging.info(f"Saved incidence results to {filename}")

    except Exception as e:
        logging.error(
            f"Error saving incidence results to NetCDF for {variable}: {e}"
        )

    finally:
        del ds
        gc.collect()


def process_variable(variable, ranges, file_paths):
    """
    Process all files for a given variable, handling both profile and incidence data.

    Parameters:
        variable (str): Variable name to process.
        ranges (list): List of bin edges.
        file_paths (list): List of file paths to process.
    """
    logging.info(f"Starting processing for variable: {variable}")
    bin_labels = generate_bin_labels(ranges)
    bin_edges = np.array(ranges)

    # Initialize accumulated results for both profile and incidence
    accumulated_results_profile = {
        str(label): {
            key: []
            for key in [
                "insitu_mask",
                "anvil_mask",
                "re_insitu",
                "re_anvil",
                "IWC_insitu",
                "IWC_anvil",
            ]
        }
        for label in bin_labels
    }

    accumulated_results_incidence = {
        str(label): {
            key: []
            for key in [
                "insitu_mask_mean",
                "anvil_mask_mean",
            ]
        }
        for label in bin_labels
    }

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(
                process_file,
                file_path,
                variable,
                bin_edges,
                bin_labels,
            )
            for file_path in file_paths
        ]
        for future in as_completed(futures):
            file_results_profile, file_results_incidence, height = (
                future.result()
            )
            if file_results_profile and file_results_incidence:
                # Accumulate profile results
                for label, data in file_results_profile.items():
                    for key, values in data.items():
                        accumulated_results_profile[label][key].append(
                            values
                        )

                # Accumulate incidence results
                for label, data in file_results_incidence.items():
                    for key, value in data.items():
                        accumulated_results_incidence[label][
                            key
                        ].append(value)

            # Clear memory after processing each file
            del file_results_profile, file_results_incidence
            gc.collect()

    # Compute final aggregated results by taking the mean
    final_results_profile = {
        label: {
            key: (np.nanmean(np.array(values), axis=0))
            for key, values in data.items()
        }
        for label, data in accumulated_results_profile.items()
    }

    final_results_incidence = {
        label: {
            key: (np.nanmean(np.array(values), axis=0))
            for key, values in data.items()
        }
        for label, data in accumulated_results_incidence.items()
    }

    # Save both profile and incidence results
    save_results_to_netcdf_profile(
        final_results_profile, OUTPUT_DIR_PROFILE, variable, height
    )
    save_results_to_netcdf_incidence(
        final_results_incidence, OUTPUT_DIR_INCIDENCE, variable
    )

    logging.info(f"Completed processing for variable: {variable}")


########################################################################################


def main():
    """Main function to orchestrate the processing of all variables."""
    deciles_dict_correct = preprocess_deciles_dict(intervals_dict)
    file_paths = sorted(
        glob.glob(os.path.join(DATA_DIR, "calced_*.nc"))
    )

    for variable, ranges in deciles_dict_correct.items():
        logging.info(f"Processing variable: {variable}")
        process_variable(variable, ranges, file_paths)

    logging.info("All processing completed.")


########################################################################################

if __name__ == "__main__":
    main()
