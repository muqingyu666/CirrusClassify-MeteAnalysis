# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2025-09-26 13:41:10
# @Last Modified by:   Muqy
# @Last Modified time: 2025-09-27 11:20:59

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import xarray as xr

from muqy_20240101_2Bgeoprof_reader import Reader


def encode_netcdf_attr(value):
    """Ensure NetCDF attributes are stored using byte-compatible types."""

    if isinstance(value, str):
        return np.bytes_(value)
    if isinstance(value, list):
        return np.array(value, dtype="S")
    return value


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
    all_file_paths: Dict[str, Dict[str, List[str]]],
    required_datasets: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Return directories containing all required datasets with aligned timestamps."""

    if required_datasets is None:
        required_datasets = {"GEOPROF"}

    common_files: Dict[str, Dict[str, List[str]]] = {}

    for day_dir, datasets in all_file_paths.items():
        if not required_datasets.issubset(datasets.keys()):
            continue

        file_identifiers = {
            dataset: {os.path.basename(f)[:13] for f in paths}
            for dataset, paths in datasets.items()
        }

        common_identifiers = set.intersection(*file_identifiers.values())

        if common_identifiers:
            common_files[day_dir] = {
                dataset: sorted(
                    [
                        f
                        for f in paths
                        if os.path.basename(f)[:13] in common_identifiers
                    ],
                    key=lambda path: os.path.basename(path)[:13],
                )
                for dataset, paths in datasets.items()
            }

    to_remove = [
        dir_name
        for dir_name, datasets in all_file_paths.items()
        if dir_name not in common_files
    ]
    for dir_name in to_remove:
        del all_file_paths[dir_name]

    return common_files


def sanitize_cloud_fraction(cf_data: np.ndarray) -> np.ndarray:
    """Clean CloudSat cloud fraction values by masking invalid entries."""

    cf_array = np.ma.filled(cf_data, np.nan).astype(np.float32)
    invalid = (cf_array < 0) | (cf_array > 100)
    cf_array[invalid] = np.nan
    return cf_array


def read_geoprof_file(file_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Read a single GEOPROF granule and extract the required variables."""

    reader = Reader(file_path)

    try:
        lon, lat, elv = reader.read_geo()
        time_index = reader.read_time(datetime=True)
        height = reader.read_sds("Height")
        height = np.ma.filled(height, np.nan).astype(np.float32)

        valid_profiles = ~np.all(np.isnan(height), axis=1)
        if not np.any(valid_profiles):
            print(f"No valid profiles in {os.path.basename(file_path)}")
            return None

        lon_values = np.asarray(lon, dtype=np.float32)[valid_profiles]
        lat_values = np.asarray(lat, dtype=np.float32)[valid_profiles]
        elv_values = np.asarray(elv, dtype=np.float32)[valid_profiles]
        time_values = time_index.to_numpy(dtype="datetime64[ns]")[
            valid_profiles
        ]

        cloud_fraction_raw = reader.read_sds("CPR_Cloud_mask", process=False)
        cloud_fraction = sanitize_cloud_fraction(cloud_fraction_raw)[
            valid_profiles
        ].astype(np.int8)

        radar_reflectivity_raw = reader.read_sds("Radar_Reflectivity")
        radar_reflectivity = np.ma.filled(
            radar_reflectivity_raw, np.nan
        ).astype(np.float32)[valid_profiles]

        height_profile = height[valid_profiles] / 1000.0  # convert to km

        return {
            "time": time_values,
            "longitude": lon_values,
            "latitude": lat_values,
            "elevation": elv_values,
            "cloud_fraction": cloud_fraction.astype(np.float32),
            "radar_reflectivity": radar_reflectivity,
            "height_profile": height_profile,
        }
    finally:
        reader.close()


def concatenate_geoprof_results(
    results: List[Dict[str, np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], np.datetime64, np.datetime64]:
    """Concatenate multiple GEOPROF granules into a single batch."""

    times = np.concatenate([res["time"] for res in results])
    longitudes = np.concatenate([res["longitude"] for res in results]).astype(
        np.float32
    )
    latitudes = np.concatenate([res["latitude"] for res in results]).astype(
        np.float32
    )
    elevations = np.concatenate([res["elevation"] for res in results]).astype(
        np.float32
    )
    cloud_fraction = np.concatenate(
        [res["cloud_fraction"] for res in results], axis=0
    ).astype(np.float32)
    radar_reflectivity = np.concatenate(
        [res["radar_reflectivity"] for res in results], axis=0
    ).astype(np.float32)
    height_profile = np.concatenate(
        [res["height_profile"] for res in results], axis=0
    ).astype(np.float32)

    sort_index = np.argsort(times)
    times = times[sort_index]
    longitudes = longitudes[sort_index]
    latitudes = latitudes[sort_index]
    elevations = elevations[sort_index]
    cloud_fraction = cloud_fraction[sort_index]
    radar_reflectivity = radar_reflectivity[sort_index]
    height_profile = height_profile[sort_index]

    height_levels = np.nanmean(height_profile, axis=0).astype(np.float32)

    return (
        {
            "time": times,
            "longitude": longitudes,
            "latitude": latitudes,
            "elevation": elevations,
            "cloud_fraction": cloud_fraction,
            "radar_reflectivity": radar_reflectivity,
            "height_profile": height_profile,
            "height_levels": height_levels,
        },
        times[0],
        times[-1],
    )


def process_directory(geoprof_paths: List[str]) -> List[Dict[str, np.ndarray]]:
    """Process all GEOPROF files within a single directory."""

    results: List[Dict[str, np.ndarray]] = []
    total = len(geoprof_paths)

    for index, file_path in enumerate(sorted(geoprof_paths)):
        print(f"[{index + 1}/{total}] Processing {os.path.basename(file_path)}")
        data = read_geoprof_file(file_path)
        if data is None:
            continue
        results.append(data)

    return results


def directory_to_date(dir_name: str) -> str:
    """Convert a YYYY/DDD directory name to YYYY_MM_DD string."""

    normalized = dir_name.replace("\\", "/")
    parts = normalized.split("/")

    if len(parts) < 2:
        raise ValueError(f"Unexpected directory format: {dir_name}")

    year, day_of_year = parts[:2]
    date_obj = datetime.strptime(f"{year}-{day_of_year}", "%Y-%j")
    return date_obj.strftime("%Y_%m_%d")


def process_and_save(
    common_paths: Dict[str, Dict[str, List[str]]],
    output_dir: str,
    dir_name: str,
) -> None:
    """Process the specified directory and persist the aggregated data."""

    geoprof_paths = common_paths[dir_name].get("GEOPROF", [])

    if not geoprof_paths:
        print(f"No GEOPROF files found for {dir_name}, skipping.")
        return

    dir_results = process_directory(geoprof_paths)

    if not dir_results:
        print(f"No valid profiles found in {dir_name}, skipping.")
        return

    data, start_time, end_time = concatenate_geoprof_results(dir_results)
    date_str = directory_to_date(dir_name)

    save_to_netcdf(
        output_dir=output_dir,
        date_str=date_str,
        data=data,
        start_time=start_time,
        end_time=end_time,
        source_files=geoprof_paths,
    )


def save_to_netcdf(
    output_dir: str,
    date_str: str,
    data: Dict[str, np.ndarray],
    start_time: np.datetime64,
    end_time: np.datetime64,
    source_files: List[str],
) -> None:
    """Save processed GEOPROF variables to a NetCDF file."""

    os.makedirs(output_dir, exist_ok=True)

    ds = xr.Dataset(
        {
            "latitude": (["time"], data["latitude"].astype(np.float32)),
            "longitude": (["time"], data["longitude"].astype(np.float32)),
            "elevation": (["time"], data["elevation"].astype(np.float32)),
            "cloud_fraction": (
                ["time", "height"],
                data["cloud_fraction"].astype(np.int8),
            ),
            "radar_reflectivity": (
                ["time", "height"],
                data["radar_reflectivity"],
            ),
            "height_profile": (
                ["time", "height"],
                data["height_profile"],
            ),
        },
        coords={
            "time": data["time"],
            "height": data["height_levels"],
        },
    )

    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["elevation"].attrs["units"] = "m"
    ds["cloud_fraction"].attrs["units"] = "%"
    ds["radar_reflectivity"].attrs["units"] = "dBZ"
    ds["height_profile"].attrs["units"] = "km"
    ds["height"].attrs["units"] = "km"

    attrs = {
        "source": "CloudSat 2B-GEOPROF",
        "start_time": np.datetime_as_string(start_time, unit="s"),
        "end_time": np.datetime_as_string(end_time, unit="s"),
        "profiles": int(data["time"].shape[0]),
        "source_files": ";".join(
            os.path.basename(file_path) for file_path in source_files
        ),
    }

    ds.attrs.update(
        {key: encode_netcdf_attr(value) for key, value in attrs.items()}
    )

    comp = dict(zlib=True, complevel=2)
    encoding = {var: comp for var in ds.data_vars}

    output_path = os.path.join(
        output_dir, f"processed_geoprof_data_{date_str}.nc"
    )

    ds.to_netcdf(output_path, mode="w", encoding=encoding, engine="h5netcdf")

    print(f"Saved {output_path}")


def extract_date_from_filename(file_name: str) -> Optional[str]:
    """Extract a YYYY_MM_DD date substring from the given filename."""

    match = re.search(r"(\d{4}_\d{2}_\d{2})", file_name)
    return match.group(1) if match else None


def find_mask_file(date_str: str, mask_dir: Path) -> Optional[Path]:
    """Locate the mask NetCDF file corresponding to the provided date string."""

    candidates = sorted(mask_dir.glob(f"*{date_str}*.nc"))
    if candidates:
        return candidates[0]


def augment_geoprof_with_masks(
    geoprof_path: Path,
    mask_path: Path,
    output_dir: Path,
    threshold: int = 15,
) -> None:
    """Split radar reflectivity by mask classification and persist the result."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(geoprof_path) as geoprof_ds, xr.open_dataset(
        mask_path
    ) as mask_ds:
        if "radar_reflectivity" not in geoprof_ds:
            print(
                f"'radar_reflectivity' not found in {geoprof_path.name}, skipping."
            )
            return

        reflectivity = (
            geoprof_ds["radar_reflectivity"].astype(np.float32).load()
        )
        insitu_mask = mask_ds["insitu_mask"].load()
        anvil_mask = mask_ds["anvil_mask"].load()

        if insitu_mask is None or anvil_mask is None:
            print(
                f"Mask dimensions do not match reflectivity for {geoprof_path.name}; skipping."
            )
            return

        insitu_condition = insitu_mask.data > threshold
        anvil_condition = anvil_mask.data > threshold

        radar_reflectivity_insitu = xr.DataArray(
            np.where(insitu_condition, reflectivity.data, np.nan),
            dims=reflectivity.dims,
            coords=reflectivity.coords,
        ).astype(np.float32)
        radar_reflectivity_anvil = xr.DataArray(
            np.where(anvil_condition, reflectivity.data, np.nan),
            dims=reflectivity.dims,
            coords=reflectivity.coords,
        ).astype(np.float32)

        augmented_ds = geoprof_ds.assign(
            insitu_mask=insitu_mask.astype(np.int16),
            anvil_mask=anvil_mask.astype(np.int16),
            radar_reflectivity_insitu=radar_reflectivity_insitu,
            radar_reflectivity_anvil=radar_reflectivity_anvil,
        )

        augmented_ds["insitu_mask"].attrs.update(
            {
                "description": "CloudSat in-situ mask",
                "threshold_applied": f"> {threshold}",
            }
        )
        augmented_ds["anvil_mask"].attrs.update(
            {
                "description": "CloudSat anvil mask",
            }
        )
        augmented_ds["radar_reflectivity_insitu"].attrs.update(
            {
                "units": "dBZ",
                "description": "Radar reflectivity where insitu_mask exceeds threshold",
            }
        )
        augmented_ds["radar_reflectivity_anvil"].attrs.update(
            {
                "units": "dBZ",
                "description": "Radar reflectivity where anvil_mask exceeds threshold",
            }
        )

        updated_attrs = dict(augmented_ds.attrs)
        updated_attrs.update(
            {
                "mask_source": mask_path.name,
                "insitu_mask_threshold": threshold,
            }
        )
        augmented_ds.attrs = {
            key: encode_netcdf_attr(value)
            for key, value in updated_attrs.items()
        }

        comp = dict(zlib=True, complevel=2)
        encoding = {var: comp for var in augmented_ds.data_vars}

        output_path = output_dir / f"{geoprof_path.stem}_mask_split.nc"

        augmented_ds.to_netcdf(
            output_path, mode="w", encoding=encoding, engine="h5netcdf"
        )

        print(
            f"Saved reflectivity split dataset: {output_path} using masks {mask_path.name}"
        )


def process_reflectivity_with_masks(
    geoprof_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    threshold: int = 15,
) -> None:
    """Iterate through GEOPROF outputs and merge with corresponding mask files."""

    if not geoprof_dir.exists() or not geoprof_dir.is_dir():
        print(
            f"GEOPROF directory {geoprof_dir} does not exist or is not a directory; skipping mask merge."
        )
        return

    mask_dir = mask_dir if mask_dir.is_dir() else mask_dir.parent
    if not mask_dir.exists():
        print(f"Mask directory {mask_dir} not found; skipping mask merge.")
        return

    geoprof_files = sorted(geoprof_dir.glob("processed_geoprof_data_*.nc"))
    if not geoprof_files:
        print(
            f"No processed GEOPROF NetCDF files found in {geoprof_dir}, skipping mask merge."
        )
        return

    for geoprof_path in geoprof_files:
        date_str = extract_date_from_filename(geoprof_path.name)
        if date_str is None:
            print(
                f"Unable to infer date from {geoprof_path.name}, skipping mask merge."
            )
            continue

        mask_path = find_mask_file(date_str, mask_dir)
        if mask_path is None:
            print(
                f"No mask NetCDF found for {date_str} in {mask_dir}, skipping."
            )
            continue

        mask_date_str = extract_date_from_filename(mask_path.name)
        if mask_date_str is None:
            print(
                f"Unable to infer date from mask {mask_path.name}; skipping mask merge for {geoprof_path.name}."
            )
            continue

        if mask_date_str != date_str:
            print(
                "Date mismatch between GEOPROF and mask files: "
                f"{geoprof_path.name} (date={date_str}) vs {mask_path.name} (date={mask_date_str}); skipping."
            )
            continue

        augment_geoprof_with_masks(
            geoprof_path=geoprof_path,
            mask_path=mask_path,
            output_dir=output_dir,
            threshold=threshold,
        )


def batch_process_geoprof(
    common_paths: Dict[str, Dict[str, List[str]]],
    output_dir: str,
) -> None:
    """Batch process GEOPROF directories sequentially."""

    directories = sorted(common_paths.keys())

    for dir_name in directories:
        print(f"Processing directory: {dir_name}")
        process_and_save(common_paths, output_dir, dir_name)


if __name__ == "__main__":
    RUN_GEOPROF_PROCESSING = False

    OUTPUT_DIR = Path(r"E:\Processed_GEOPROF_data")
    MASK_DIR = Path(
        r"Q:\Calced_Vars_Cirrus_classification_3000m_overlay_NEW_aerosol_0085"
    )
    REFLECTIVITY_OUTPUT_DIR = Path(r"E:\Processed_GEOPROF_data_with_masks")
    REFLECTIVITY_THRESHOLD = 15

    base_paths = {
        "CLDCLASS": r"P:\CLOUDSAT_DATA\2B_CLDCLASS_LIDAR",
        "ICE": r"P:\CLOUDSAT_DATA\2C_ICE",
        "GEOPROF_LIDAR": r"P:\CLOUDSAT_DATA\2B_GEOPROF_LIDAR",
        "GEOPROF": r"E:\2B_GEOPROF",
        "ECMWF_AUX": r"G:\Data\CLOUDSAT_DATA\ECMWF_AUX",
        "MODIS_AUX": r"D:\MODIS_AUX",
    }

    all_file_paths = get_all_files_by_directory(base_paths)

    common_paths = find_common_dirs(
        all_file_paths,
        required_datasets={
            "CLDCLASS",
            "ICE",
            "GEOPROF",
            "ECMWF_AUX",
            "GEOPROF_LIDAR",
        },
    )

    if RUN_GEOPROF_PROCESSING:
        if not common_paths:
            print("No GEOPROF directories found. Please verify the base paths.")
        else:
            batch_process_geoprof(common_paths, str(OUTPUT_DIR))
    else:
        print(
            "Skipping GEOPROF aggregation step (RUN_GEOPROF_PROCESSING=False)."
        )

    process_reflectivity_with_masks(
        geoprof_dir=OUTPUT_DIR,
        mask_dir=MASK_DIR,
        output_dir=REFLECTIVITY_OUTPUT_DIR,
        threshold=REFLECTIVITY_THRESHOLD,
    )
