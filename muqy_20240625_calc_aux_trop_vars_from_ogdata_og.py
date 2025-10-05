# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:44:44

import gc
import glob
import os
import time
from multiprocessing import Pool

import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from metpy.units import units
import warnings
from datetime import datetime

# Disable all warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# Functions to extract dates from filenames


def extract_start_date(filename):
    """
    Extracts the start date from a filename.
    """
    date_part = filename.split("/")[-1].split("_")[-3:]
    date_part[-1] = date_part[-1].split(".")[0]
    date_str = "_".join(date_part)
    return datetime.strptime(date_str, "%Y_%m_%d")


def extract_dates(filename):
    """
    Extracts the start and end dates from a filename.
    """
    # extract the date parts from the filename
    date_parts = filename.split("/")[-1].split("_")[3:9]
    # delete ".nc" from the last part
    date_parts[-1] = date_parts[-1].split(".")[0]
    # join the date parts to form the start and end dates
    start_date_str = "_".join(date_parts[:3])
    end_date_str = "_".join(date_parts[3:])
    return start_date_str, end_date_str


########################################################################################
##### Functions to calculate tropospheric variables ####################################
########################################################################################


def find_closest_index(height_array, target_height):
    """
    Find the index of the closest value in the height_array to the target_height.

    Parameters:
    height_array (numpy.ndarray): An array of heights.
    target_height (float): The target height.

    Returns:
    int: The index of the closest value in the height_array to the target_height.
    """
    return np.nanargmin(np.abs(height_array - target_height))


def calculate_potential_temperature(temperature, pressure):
    """
    Calculate the potential temperature.
    """
    # Calculate potential temperature
    potential_temperature = mpcalc.potential_temperature(
        pressure * units.hPa, temperature * units.kelvin
    )
    return potential_temperature.magnitude


def calculate_equivalent_potential_temperature(temperature, pressure, dewpoint):
    """
    Calculate the potential temperature.
    """

    # Calculate equivalent potential temperature
    equivalent_potential_temperature = mpcalc.equivalent_potential_temperature(
        pressure * units.hPa,
        temperature * units.kelvin,
        dewpoint,
    )

    return equivalent_potential_temperature.magnitude


def calculate_lapse_rate(temperature, height):
    """
    Calculate the lapse rate.
    """
    # Calculate the lapse rate
    lapse_rate = np.gradient(temperature, height, axis=1)
    # Units: K/km
    return lapse_rate


def calculate_potential_temperature_gradient(potential_temperature, height):
    """
    Calculate the lapse rate.
    """
    # Calculate the lapse rate
    potential_temperature_gradient = np.gradient(
        potential_temperature, height, axis=1
    )
    # Units: K/km
    return potential_temperature_gradient


def calculate_relative_humidity(pressure, temperature, specific_humidity):
    relative_humidity = mpcalc.relative_humidity_from_specific_humidity(
        pressure * units.hPa,
        temperature * units.kelvin,
        specific_humidity * units("kg/kg"),
    ).to("percent")

    return relative_humidity.magnitude


########################################################################################
# Old and new definitions of the tropopause height, just for reference
########################################################################################


def find_tropopause_old(lapse_rate, height):
    """
    Find the tropopause height for each contour.
    Old definition of the tropopause.
    1957 WMO definition of the tropopause:
        The lowest level at which the lapse rate decreases to 2 K/km or less,
        provided that the average lapse rate between this level and
        all higher levels within 2 km does not exceed 2 K/km.
    """
    tropopause_height = np.full(lapse_rate.shape[0], np.nan)

    for time in range(lapse_rate.shape[0]):
        for hgt in range(lapse_rate.shape[1] - 1):
            # If the lapse rate is less than 2 K/km
            if lapse_rate[time, hgt] < 2:
                # Find the index of the closest value in the height_array to height[hgt] + 2km
                hgt_idx = find_closest_index(height, height[hgt] + 2)
                avg_lapse_rate = np.nanmean(lapse_rate[time, hgt_idx:hgt])
                if avg_lapse_rate > 2:
                    continue
                if avg_lapse_rate <= 2:
                    tropopause_height[time] = height[hgt]
                    break

    return tropopause_height


def find_tropopause_new(potential_temperature_gradient, heights):
    """
    Find the first and second tropopause height and temperature.
        Its the new and perhaps better definition of the tropopause.
        Definitions from A Modern Approach to a Stability-Based Deﬁnition of the Tropopause by EMILY N. TINNEY 2022
        doi: 10.1175/MWR-D-22-0174.1
        published on Monthly Weather Review
    """
    tropopause_height = np.full(potential_temperature_gradient.shape[0], np.nan)

    # Loop over time/profiles to find the tropopause
    for time in range(potential_temperature_gradient.shape[0]):
        # Find first tropopause
        for hgt_index in range(
            potential_temperature_gradient.shape[1] - 1, -1, -1
        ):
            # If the potential temperature gradient is greater than 10 K/km
            if potential_temperature_gradient[time, hgt_index] >= 10:

                hgt_index_2km_higher = find_closest_index(
                    heights, heights[hgt_index] + 2
                )

                avg_gradient = np.mean(
                    potential_temperature_gradient[
                        time, hgt_index : hgt_index_2km_higher + 2
                    ]
                )

                # If the average gradient 2km above this level is less than 10 K/km
                if avg_gradient < 10:
                    continue

                # If the average gradient 2km above this level is greater than 10 K/km
                if avg_gradient > 10:
                    tropopause_height[time] = heights[hgt_index]
                    break

    return tropopause_height


########################################################################################
# Packed parallelized version of the new tropopause height calculation
########################################################################################


def process_profile_tropopause(gradients, heights):
    """
    Process a single profile to find the tropopause height.
    """
    num_heights = len(heights)

    # Find tropopause
    for hgt_index in range(num_heights - 1, -1, -1):

        # If the potential temperature gradient is greater than 10.5 K/km and heights > 5km
        if gradients[hgt_index] > 10.5 and heights[hgt_index] > 4:
            target_height = heights[hgt_index] + 2
            hgt_index_2km_higher = find_closest_index(heights, target_height)

            if hgt_index_2km_higher == 0 and hgt_index == 0:
                avg_gradient = np.nanmean(
                    gradients[hgt_index : hgt_index_2km_higher + 2]
                )
            else:
                avg_gradient = np.nanmean(
                    gradients[hgt_index_2km_higher:hgt_index]
                )

            # If the average gradient 2km above this level is greater than 10 K/km
            if avg_gradient > 10:
                return heights[hgt_index]

    return np.nan


def find_tropopause_new_parallel(
    potential_temperature_gradient, heights, n_jobs=-1
):
    """
    Find the first and second tropopause height and temperature.
    Parallelized version.
    """
    num_profiles = potential_temperature_gradient.shape[0]

    # Process profiles in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_profile_tropopause)(
            potential_temperature_gradient[time_idx], heights
        )
        for time_idx in range(num_profiles)
    )

    return np.array(results, dtype=np.float32)


########################################################################################


def calc_trop_vars(
    lats,
    lons,
    insitu_mask,
    anvil_mask,
    DCS_mask,
    pressure,
    temperature,
    specific_humidity,
    u_wind,
    v_wind,
    skin_temperature,
    time,
    heights,
    elvs,
    re,
    IWC,
):
    """
    Calculates various tropospheric variables based on given input atmospheric profiles.
    """
    # ----------------------------------------------------------------------------------
    # Classify microphysics according to cirrus mask
    re_insitu = np.where(insitu_mask > 20, re, np.nan)
    re_anvil = np.where(anvil_mask > 20, re, np.nan)
    IWC_insitu = np.where(insitu_mask > 20, IWC, np.nan)
    IWC_anvil = np.where(anvil_mask > 20, IWC, np.nan)
    # ----------------------------------------------------------------------------------
    # Calculate derived thermodynamic variables
    dewpoint = mpcalc.dewpoint_from_specific_humidity(
        pressure * units.hPa,
        temperature * units.kelvin,
        specific_humidity * units("kg/kg"),
    )
    potential_temp = calculate_potential_temperature(temperature, pressure)
    equivalent_potential_temp = calculate_equivalent_potential_temperature(
        temperature, pressure, dewpoint
    )
    potential_temp_grad = calculate_potential_temperature_gradient(
        potential_temp, heights
    )
    tropopause_height = find_tropopause_new_parallel(
        potential_temp_grad, heights
    )
    relative_humidity = calculate_relative_humidity(
        pressure, temperature, specific_humidity
    )

    # ----------------------------------------------------------------------------------
    # Initialize arrays for results
    num_profiles = len(tropopause_height)
    nan_array = np.full(num_profiles, np.nan, dtype=np.float32)

    results = {
        "lats": lats,
        "lons": lons,
        "time": time,
        "elvs": elvs,
        "insitu_mask": insitu_mask,
        "anvil_mask": anvil_mask,
        "DCS_mask": DCS_mask,
        "re_insitu": re_insitu,
        "re_anvil": re_anvil,
        "IWC_insitu": IWC_insitu,
        "IWC_anvil": IWC_anvil,
        "tropopause_height": tropopause_height.astype(np.float32),
        "heights": heights,
        "upper_tropopause_stability": nan_array.copy(),
        # "instability": nan_array.copy(),
        "tropopause_temp": nan_array.copy(),
        # "upper_trop_temp": nan_array.copy(),
        "tropopause_u_wind": nan_array.copy(),
        # "upper_trop_u_wind": nan_array.copy(),
        "tropopause_v_wind": nan_array.copy(),
        # "upper_trop_v_wind": nan_array.copy(),
        "skin_temperature": nan_array.copy(),
        "tropopause_humidity": nan_array.copy(),
        # "upper_trop_humidity": nan_array.copy(),
        "upper_tropopause_wind_shear": nan_array.copy(),
    }

    # ----------------------------------------------------------------------------------
    # Loop through profiles to compute variables
    for idx, top_height in enumerate(tropopause_height):
        if np.isnan(top_height):
            continue

        top_index = find_closest_index(heights, top_height)
        pressure_200_below = pressure[idx, top_index] + 200
        idx_200_below = np.nanargmin(np.abs(pressure[idx] - pressure_200_below))

        # Stability and instability
        height_diff = heights[top_index] - heights[idx_200_below]
        if height_diff > 0:
            results["upper_tropopause_stability"][idx] = (
                potential_temp[idx, top_index]
                - potential_temp[idx, idx_200_below]
            ) / height_diff
            # results["instability"][idx] = (
            #     equivalent_potential_temp[idx, top_index]
            #     - equivalent_potential_temp[idx, idx_200_below]
            # ) / height_diff

        # Tropopause and upper troposphere variables
        results["tropopause_humidity"][idx] = relative_humidity[idx, top_index]
        # results["upper_trop_humidity"][idx] = np.mean(
        #     relative_humidity[idx, top_index:idx_200_below]
        # )
        results["tropopause_temp"][idx] = temperature[idx, top_index]
        # results["upper_trop_temp"][idx] = np.mean(
        #     temperature[idx, top_index:idx_200_below]
        # )
        results["skin_temperature"][idx] = skin_temperature[idx]
        results["tropopause_u_wind"][idx] = u_wind[idx, top_index]
        # results["upper_trop_u_wind"][idx] = np.mean(
        #     u_wind[idx, top_index:idx_200_below]
        # )
        results["tropopause_v_wind"][idx] = v_wind[idx, top_index]
        # results["upper_trop_v_wind"][idx] = np.mean(
        #     v_wind[idx, top_index:idx_200_below]
        # )

        # Wind shear
        results["upper_tropopause_wind_shear"][idx] = (
            u_wind[idx, top_index] - u_wind[idx, idx_200_below]
        ) / height_diff

    # ----------------------------------------------------------------------------------
    # Clean up and return results
    del potential_temp, equivalent_potential_temp, potential_temp_grad
    gc.collect()

    return results


def save_to_netcdf(input_file_path, output_file_path, data):

    ds = xr.Dataset(
        {
            "latitude": (
                ["time"],
                data["lats"],
                {"dtype": "float32"},
            ),
            "longitude": (
                ["time"],
                data["lons"],
                {"dtype": "float32"},
            ),
            "elevation": (
                ["time"],
                data["elvs"],
                {"dtype": "float32"},
            ),
            "insitu_mask": (
                ["time", "height"],
                data["insitu_mask"],
                {"dtype": "int8"},
            ),
            "anvil_mask": (
                ["time", "height"],
                data["anvil_mask"],
                {"dtype": "int8"},
            ),
            "DCS_mask": (
                ["time", "height"],
                data["DCS_mask"],
                {"dtype": "int8"},
            ),
            "re_insitu": (
                ["time", "height"],
                data["re_insitu"],
                {"dtype": "float32"},
            ),
            "re_anvil": (
                ["time", "height"],
                data["re_anvil"],
                {"dtype": "float32"},
            ),
            "IWC_insitu": (
                ["time", "height"],
                data["IWC_insitu"],
                {"dtype": "float32"},
            ),
            "IWC_anvil": (
                ["time", "height"],
                data["IWC_anvil"],
                {"dtype": "float32"},
            ),
            "Tropopause_height": (
                ["time"],
                data["tropopause_height"],
                {"dtype": "float32"},
            ),
            "Upper_tropopause_stability": (
                ["time"],
                data["upper_tropopause_stability"],
                {"dtype": "float32"},
            ),
            # "Instability": (
            #     ["time"],
            #     data["instability"],
            # ),
            "Tropopause_relative_humidity": (
                ["time"],
                data["tropopause_humidity"],
                {"dtype": "float32"},
            ),
            # "Upper_trop_humidity": (
            #     ["time"],
            #     data["upper_trop_humidity"],
            # ),
            "Tropopause_u_wind": (
                ["time"],
                data["tropopause_u_wind"],
                {"dtype": "float32"},
            ),
            # "Upper_trop_u_wind": (
            #     ["time"],
            #     data["upper_trop_u_wind"],
            #     {"dtype": "float32"},
            # ),
            "Tropopause_v_wind": (
                ["time"],
                data["tropopause_v_wind"],
                {"dtype": "float32"},
            ),
            # "Upper_trop_v_wind": (
            #     ["time"],
            #     data["upper_trop_v_wind"],
            # ),
            "Upper_tropopause_wind_shear": (
                ["time"],
                data["upper_tropopause_wind_shear"],
                {"dtype": "float32"},
            ),
            "Skin_temperature": (
                ["time"],
                data["skin_temperature"],
                {"dtype": "float32"},
            ),
            # "Upper_trop_temp": (
            #     ["time"],
            #     data["upper_trop_temp"],
            # ),
            "Tropopause_temp": (
                ["time"],
                data["tropopause_temp"],
                {"dtype": "float32"},
            ),
        },
        coords={"height": data["heights"], "time": data["time"]},
    )

    # Define file name
    file_path = output_file_path + input_file_path.split("/")[-1].replace(
        "processed", "calced_vars"
    )

    # Define compression options
    comp = dict(zlib=True, complevel=1)  # Adjust complevel as needed

    # Apply compression to all variables
    encoding = {var: comp for var in ds.data_vars}

    # Save the dataset to a NetCDF file
    ds.to_netcdf(file_path, mode="w", engine="h5netcdf", encoding=encoding)

    # Print the output file path
    print(f"Saved {file_path}")


def process_calculation_and_save(input_file_path, output_file_path):

    # Open each input file to extract the necessary variables
    with xr.open_dataset(input_file_path, engine="h5netcdf") as ds:
        # List of variables to process
        met_vars = [
            "pressure",
            "temperature",
            "specific_humidity",
            "u_wind",
            "v_wind",
            "skin_temperature",
        ]

        other_vars = [
            "height",
            "latitude",
            "longitude",
            "insitu_mask",
            "anvil_mask",
            "DCS_mask",
            "elevation",
            "re",
            "IWC",
        ]

        # Process meteorological variables with NaN conversion and float32
        processed_data = {}
        for var in met_vars:
            data = ds[var].values.astype(np.float32)
            np.putmask(data, data <= -999, np.nan)
            processed_data[var] = data

        # Process other variables without conversion
        for var in other_vars:
            processed_data[var] = ds[var].values

        # Handle time separately
        processed_data["time"] = ds["time"].values

    # Calculate tropospheric variables
    trop_vars = calc_trop_vars(
        processed_data["latitude"],
        processed_data["longitude"],
        processed_data["insitu_mask"],
        processed_data["anvil_mask"],
        processed_data["DCS_mask"],
        processed_data["pressure"],
        processed_data["temperature"],
        processed_data["specific_humidity"],
        processed_data["u_wind"],
        processed_data["v_wind"],
        processed_data["skin_temperature"],
        processed_data["time"],
        processed_data["height"],
        processed_data["elevation"],
        processed_data["re"],
        processed_data["IWC"],
    )

    # Save the calculated variables to a new NetCDF file
    save_to_netcdf(input_file_path, output_file_path, trop_vars)

    del trop_vars, processed_data
    gc.collect()


def main_execution(input_file_paths, output_file_path, num_workers=6):

    # Make the output directory if it does not exist
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        pool.starmap(
            process_calculation_and_save,
            [
                (input_file_path, output_file_path)
                for input_file_path in input_file_paths
            ],
        )

    print(f"Total time: {time.time() - start_time}")


########################################################################################


if __name__ == "__main__":
    pass


########################################################################################
