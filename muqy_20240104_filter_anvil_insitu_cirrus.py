# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:44:09

import fnmatch
import os
import warnings
import gc
import matplotlib.colorbar as colorbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from muqy_20240101_2Bgeoprof_reader import Reader
from muqy_20240101_plot_2Bgeoprof_test import draw_cross_section
from muqy_20240102_plot_2Bcldtype_2Bcldfrac_2Cice import (
    draw_cloud_profile,
    draw_cross_section,
    draw_elevation,
    filter_by_time_range,
)
from scipy.ndimage import binary_dilation, label

# Ignore all warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Initiate Satellite data Processor
# ----------------------------------------------------------------


def get_all_files(base_path):
    """
    Get a list of all file paths matching the pattern '?????????????_*.hdf'
    within the specified base path and its subdirectories.

    Args:
        base_path (str): The base path to search for files.

    Returns:
        list: A list of file paths matching the pattern.
    """
    file_paths = []
    for root, _, files in os.walk(base_path):
        for file in fnmatch.filter(files, "?????????????_?????_*.hdf"):
            file_paths.append(os.path.join(root, file))
    return file_paths


def find_common_files(all_file_paths):
    """
    Find the common files among multiple sets of file paths.

    Parameters:
    all_file_paths (dict): A dictionary containing sets of file paths.

    Returns:
    dict: A dictionary containing the common files for each set of file paths.
    """
    file_identifiers = {
        key: set(os.path.basename(f)[:13] for f in paths)
        for key, paths in all_file_paths.items()
    }
    common_identifiers = set.intersection(*file_identifiers.values())

    common_files = {
        key: sorted(
            [
                f
                for f in paths
                if os.path.basename(f)[:13] in common_identifiers
            ],
            key=lambda x: os.path.basename(x)[:13],
        )
        for key, paths in all_file_paths.items()
    }

    return common_files


def find_and_count_common_files(base_paths):
    """
    Find and count common files among multiple base paths.

    Args:
        base_paths (dict): A dictionary containing base paths as keys and corresponding paths as values.

    Returns:
        dict: A dictionary containing common files found among the base paths.

    """
    all_file_paths = {
        key: get_all_files(path) for key, path in base_paths.items()
    }
    common_files = find_common_files(all_file_paths)

    # Count the number of common files
    common_file_count = (
        sum(len(files) for files in common_files.values()) // len(common_files)
        if common_files
        else 0
    )
    print(f"Found {common_file_count} common files")

    return common_files, common_file_count


# Main code to run the cirrus classification
class CloudSatProcessor_Ver2:
    def __init__(
        self,
        common_files,
        structure_0=np.ones((7, 1)),
        structure_1=np.ones((1, 7)),
    ):
        self.common_files = common_files
        self.data = {}
        self.aux_cld_type = None
        self.aux_cloud_labeled = None
        self.structure_0 = structure_0
        self.structure_1 = structure_1

    def read_data(self, file_index=0, dataset_names=None):
        """
        Reads data from the specified dataset names and file index.

        Args:
            file_index (int): The index of the file to read from. Default is 0.
            dataset_names (list): A list of dataset names to read from.

        Returns:
            None

        Raises:
            None
        """

        if dataset_names is None:
            print("No dataset specified.")
            return

        for dataset_name in dataset_names:
            if dataset_name not in self.common_files:
                print(f"Dataset '{dataset_name}' not found in common files.")
                continue

            files = self.common_files[dataset_name]

            if 0 <= file_index < len(files):
                data = self.read_cloudsat_data(files[file_index])
                self.data.update(data)
            else:
                print(
                    f"No file at index {file_index} for dataset '{dataset_name}'"
                )

    def process_cloud_fraction(self, cf_data):
        """
        Adjusts cloud fraction data based on specified conditions.
        If cloud fraction data is greater than or equal to 50, assign it to 1,
        otherwise, assign it to 0.

        Parameters:
        cf_data (numpy.ndarray): Array of cloud fraction data.

        Returns:
        numpy.ndarray: Array where each element is 1 if original value >= 50, otherwise 0.
        """
        return np.where((cf_data < 0) | (cf_data > 100), -1, cf_data).astype(
            np.int8
        )

    def read_cloudsat_data(self, file_path):
        """
        Read CloudSat data from the specified file.

        Args:
            file_path (str): The path to the CloudSat data file.

        Returns:
            dict: A dictionary containing the CloudSat data.
        """
        f = Reader(file_path)
        data = {}
        if "CLDCLASS" in file_path:
            data = self.process_cldclass(f)
        elif "ICE" in file_path:
            data = self.process_ice(f)
        elif "GEOPROF" in file_path:
            data = self.process_geoprof(f)
        elif "ECMWF_AUX" in file_path:
            data = self.process_ec_aux(f)
        elif "MODIS_AUX" in file_path:
            data = self.process_modis(f)
        else:
            print(f"No processor defined for dataset '{file_path}'")
            return

        return data

    def process_cldclass(self, f):
        lon, lat, elv = f.read_geo()
        time, datetime_data = f.read_time(datetime=False), f.read_time(
            datetime=True
        )
        height = f.read_sds("Height")
        self.valid_profiles = (~np.all(np.isnan(height), axis=1),)

        return {
            "lon": lon[self.valid_profiles],
            "lat": lat[self.valid_profiles],
            "elv": elv[self.valid_profiles],
            "time": time[self.valid_profiles],
            "start_time": datetime_data[0],
            "end_time": datetime_data[-1],
            "cld_layer_base": f.read_sds("CloudLayerBase")[self.valid_profiles],
            "cld_layer_top": f.read_sds("CloudLayerTop")[self.valid_profiles],
            "cld_layer_type": f.read_sds("CloudLayerType")[self.valid_profiles],
        }

    def process_modis(self, f):
        return {
            "Solar_zenith": np.mean(
                f.read_sds("Solar_zenith", process=False), axis=1
            ).astype(np.int16)[self.combined_filter],
            "Solar_azimuth": np.mean(
                f.read_sds("Solar_azimuth", process=False), axis=1
            ).astype(np.int16)[self.combined_filter],
        }

    def process_ice(self, f):
        return {
            "re": f.read_sds("re")[self.valid_profiles],
            "IWC": f.read_sds("IWC", process=False)[self.valid_profiles],
            "EXT_coef": f.read_sds("EXT_coef", process=False)[
                self.valid_profiles
            ],
        }

    def process_geoprof(self, f):
        cloud_fraction = f.read_sds("CloudFraction", process=False)
        height = f.read_sds("Height")
        return {
            "cld_frac": self.process_cloud_fraction(cloud_fraction)[
                self.valid_profiles
            ].astype(np.float32),
            "height": height[self.valid_profiles],
            "hgt": np.nanmean(height[self.valid_profiles], axis=0) / 1000,
        }

    def process_ec_aux(self, f):
        return {
            "temperature": f.read_sds("Temperature", process=False)[
                self.valid_profiles
            ],
            "specific_humidity": f.read_sds("Specific_humidity", process=False)[
                self.valid_profiles
            ],
            "u_velocity": f.read_sds("U_velocity", process=False)[
                self.valid_profiles
            ],
            "v_velocity": f.read_sds("V_velocity", process=False)[
                self.valid_profiles
            ],
            "pressure": f.read_sds("Pressure", process=False)[
                self.valid_profiles
            ]
            / 100,
            "skin_temperature": np.array(
                f.attach_vdata("Skin_temperature")
            ).flatten()[self.valid_profiles],
        }

    def create_aux_cld_data(self, required_types=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        """
        Create auxiliary cloud data based on the given required cloud types.

        Parameters:
            required_types (list): List of required cloud types.
            Default is [0, 1, 2, 3, 4, 5, 6, 7, 8].

        Returns:
            None
        """
        cloud_type = self.data["cld_layer_type"]
        cloud_base = self.data["cld_layer_base"]
        cloud_top = self.data["cld_layer_top"]
        hgt = self.data["hgt"]

        type_mask = np.isin(cloud_type, required_types)
        type_mask[np.isnan(cloud_type)] = 0
        filtered_cloud_type = np.where(type_mask, cloud_type, 0).astype(np.int8)
        filtered_cloud_base = np.where(type_mask, cloud_base, np.nan).astype(
            np.float32
        )
        filtered_cloud_top = np.where(type_mask, cloud_top, np.nan).astype(
            np.float32
        )

        # Create a new array to store the auxiliary cloud data
        aux_cld_type = np.full_like(self.data["cld_frac"], 0).astype(np.int8)

        # # Loop through each cloud layer
        # Vectorized approach to replace nested for-loops
        for ncloud in range(cloud_type.shape[1]):
            # Create a mask for the current cloud layer base
            base_height = np.repeat(
                filtered_cloud_base[:, ncloud, np.newaxis],
                hgt.shape[0],
                axis=1,
            )
            # Create a mask for the current cloud layer top
            top_height = np.repeat(
                filtered_cloud_top[:, ncloud, np.newaxis],
                hgt.shape[0],
                axis=1,
            )

            # Create a mask for the current cloud layer
            within_layer = (base_height <= hgt) & (hgt <= top_height)

            # Update the auxiliary cloud data
            aux_cld_type[within_layer] = np.repeat(
                filtered_cloud_type[:, ncloud, np.newaxis],
                hgt.shape[0],
                axis=1,
            )[within_layer]

        # Update the attribute
        self.aux_cld_type = aux_cld_type
        self.filtered_cloud_type = filtered_cloud_type.astype(np.int8)

    def apply_connected_component_labeling(self):
        """
        Apply connected component labeling to the auxiliary cloud data.

        This method assigns labels to connected components in the auxiliary cloud data.
        It replaces NaN values with 0 and non-NaN values with 1 before applying the labeling logic.

        After labeling, the method updates the `aux_cloud_labeled` attribute with the labeled data.
        The actual number of features is assigned to the `num_features` attribute.

        Returns:
            None
        """
        # now all cloud bins are labeled as 1, and non-cloud bins are labeled as 0
        # apply connected component labeling method
        # https://en.wikipedia.org/wiki/Connected-component_labeling

        # Convert NaN to 0 and non-NaN to 1, then apply labeling
        binary_cloud_data = np.where(self.aux_cld_type == 0, 0, 1)
        dilated_binary_cloud_data = binary_dilation(
            binary_cloud_data, structure=np.ones((2, 1))
        )
        # dilated_binary_cloud_data = binary_cloud_data
        labeled_cloud_data, num_features = label(
            dilated_binary_cloud_data, structure=np.ones((3, 3))
        )

        self.binary_cloud_data = binary_cloud_data
        self.aux_cloud_labeled = labeled_cloud_data
        self.num_features = num_features

    def filter_cloud_clusters_connected_to_cirrus_or_DC(self):
        """
        Filters cloud clusters connected to cirrus or deep convection.

        Returns:
            tuple: (bool, bool) indicating presence of anvil and in-situ cirrus
        """
        # Pre-compute masks once
        cirrus_mask = self.aux_cld_type == 1
        dc_mask = self.aux_cld_type == 8

        # Use more efficient data types
        anvil_cirrus_mask = np.zeros_like(self.aux_cld_type, dtype=np.int8)
        aux_anvil_cirrus_mask = np.zeros_like(self.aux_cld_type, dtype=np.int16)
        insitu_cirrus_mask = np.zeros_like(self.aux_cld_type, dtype=np.int8)
        aux_insitu_cirrus_mask = np.zeros_like(
            self.aux_cld_type, dtype=np.int16
        )

        # Process clusters in parallel with optimized chunk size
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(self._process_cluster)(i, cirrus_mask, dc_mask)
            for i in range(self.num_features)
        )

        anvil_cirrus_idx = insitu_cirrus_idx = 0

        # Vectorized processing of results
        for (
            only_cirrus,
            contains_dc_and_cirrus,
            cloud_cluster_mask,
        ) in results:
            if contains_dc_and_cirrus:
                cirrus_subset = cloud_cluster_mask & cirrus_mask
                anvil_cirrus_mask[cirrus_subset] = self.aux_cld_type[
                    cirrus_subset
                ]
                anvil_cirrus_idx += 1
                aux_anvil_cirrus_mask[cirrus_subset] = anvil_cirrus_idx

            if only_cirrus:
                insitu_cirrus_mask[cloud_cluster_mask] = self.aux_cld_type[
                    cloud_cluster_mask
                ]
                insitu_cirrus_idx += 1
                aux_insitu_cirrus_mask[cloud_cluster_mask] = insitu_cirrus_idx

        # data = self.data.copy()
        # data["cld_frac"] = insitu_cirrus_mask
        # processor.filter_and_plot_data_aux_simple_1fig(
        #     data,
        #     start_time_idx_manual=start_time_idx,
        #     time_range_custum_manual=time_range_custum,
        #     subplot_2_title="Initial",
        #     subplot_2_clable="Cloud Cluster ID",
        # )

        # Update instance variables efficiently
        self.anvil_cirrus_mask = anvil_cirrus_mask
        self.aux_anvil_cirrus_mask = aux_anvil_cirrus_mask
        self.anvil_cirrus_idx = anvil_cirrus_idx
        self.insitu_cirrus_mask = insitu_cirrus_mask
        self.aux_insitu_cirrus_mask = aux_insitu_cirrus_mask
        self.insitu_cirrus_idx = insitu_cirrus_idx

        return anvil_cirrus_idx > 0, insitu_cirrus_idx > 0

    def _process_cluster(self, cloud_cluster_idx, cirrus_mask, dc_mask):
        """
        Process individual cloud clusters efficiently.

        Args:
            cloud_cluster_idx (int): Index of the cloud cluster
            cirrus_mask (ndarray): Pre-computed cirrus mask
            dc_mask (ndarray): Pre-computed deep convection mask

        Returns:
            tuple: (only_cirrus, contains_dc_and_cirrus, cloud_cluster_mask)
        """
        cloud_cluster_mask = self.aux_cloud_labeled == cloud_cluster_idx

        # Vectorized operations for cluster analysis
        cluster_types = self.aux_cld_type[cloud_cluster_mask]
        only_cirrus = np.any(cluster_types == 1) and np.all(
            np.isin(cluster_types, [0, 1])
        )
        contains_cirrus = np.any(cirrus_mask[cloud_cluster_mask])
        contains_dc = np.any(dc_mask[cloud_cluster_mask])

        return (
            only_cirrus,
            (contains_cirrus and contains_dc),
            cloud_cluster_mask,
        )

    def iterative_expand_anvil_cirrus(
        self,
        iterations=25,
        start_time_idx=None,
        time_range_custum=None,
    ):

        self.extended_anvil_cirrus_mask = np.zeros_like(self.anvil_cirrus_mask)

        # Iteratively expand the anvil cirrus clouds
        for iteration in range(iterations):
            # Initialize temporary variables
            clusters = np.copy(self.anvil_cirrus_mask)

            # data = self.data.copy()
            # data["cld_frac"] = clusters
            # processor.filter_and_plot_data_aux_simple_1fig(
            #     data,
            #     start_time_idx_manual=start_time_idx,
            #     time_range_custum_manual=time_range_custum,
            #     subplot_2_title="Initial",
            #     subplot_2_clable="Cloud Cluster ID",
            # )

            Parallel(n_jobs=1, backend="threading")(
                delayed(self._process_anvil_cirrus_cluster)(
                    # dilated anvil mask
                    iteration,
                    cloud_cluster_idx,
                )
                for cloud_cluster_idx in range(1, self.anvil_cirrus_idx + 1)
            )

            # Check if the anvil cirrus clouds have converged
            if np.array_equal(clusters, self.anvil_cirrus_mask):
                # print(f"Converged after {iteration+1} iterations.")
                break

    def _dilate_mask_inner(
        self,
        clusters,
    ):
        dilated_mask = binary_dilation(clusters, structure=self.structure_0)
        dilated_mask = binary_dilation(dilated_mask, structure=self.structure_1)
        return dilated_mask

    def _process_anvil_cirrus_cluster(
        self,
        iteration,
        anvil_cirrus_idx,
    ):

        # The current generation of anvil cirrus clouds
        current_anvil_cirrus_mask = (
            self.aux_anvil_cirrus_mask == anvil_cirrus_idx
        )
        # data = self.data.copy()
        # data["cld_frac"] = current_anvil_cirrus_mask
        # processor.filter_and_plot_data_aux_simple_1fig(
        #     data,
        #     start_time_idx_manual=start_time_idx,
        #     time_range_custum_manual=time_range_custum,
        #     subplot_2_title="Current",
        #     subplot_2_clable="Cloud Cluster ID",
        # )

        dilated_current_anvil_cirrus_mask = self._dilate_mask_inner(
            current_anvil_cirrus_mask
        )

        # data = self.data.copy()
        # data["cld_frac"] = dilated_current_anvil_cirrus_mask
        # processor.filter_and_plot_data_aux_simple_1fig(
        #     data,
        #     start_time_idx_manual=start_time_idx,
        #     time_range_custum_manual=time_range_custum,
        #     subplot_2_title="Dilated",
        #     subplot_2_clable="Cloud Cluster ID",
        # )

        if iteration == 0:
            self.extended_anvil_cirrus_mask[current_anvil_cirrus_mask] = 1

        if np.any(
            self.insitu_cirrus_mask[dilated_current_anvil_cirrus_mask] == 1
        ):

            # The insitu cirrus number we captured
            insitu_cirrus_idxs = np.unique(
                self.aux_insitu_cirrus_mask[dilated_current_anvil_cirrus_mask]
            )
            # Delete the 0 index
            insitu_cirrus_idxs = insitu_cirrus_idxs[insitu_cirrus_idxs != 0]

            # Loop through every insitu cirrus we captured to determine if we should extend the anvil cirrus
            # Based on IWC
            for insitu_cirrus_idx in insitu_cirrus_idxs:

                # Current insitu cirrus mask
                current_insitu_cirrus_mask = (
                    self.aux_insitu_cirrus_mask == insitu_cirrus_idx
                )

                # Determine if the current insitu cirrus cloud layer has a lower IWC average and maximum
                # Than the last generation of anvil cirrus clouds
                if np.any(current_insitu_cirrus_mask) and np.any(
                    current_anvil_cirrus_mask
                ):
                    # if np.nanmean(
                    #     self.data["IWC"][current_insitu_cirrus_mask]
                    # ) <= np.nanmean(
                    #     self.data["IWC"][
                    #         self.extended_anvil_cirrus_mask
                    #         == iteration + 1
                    #     ]

                    # ) and np.nanmax(
                    #     self.data["IWC"][current_insitu_cirrus_mask]
                    # ) <= np.nanmax(
                    #     self.data["IWC"][
                    #         self.extended_anvil_cirrus_mask
                    #         == iteration + 1
                    #     ]

                    # ):
                    # print(
                    #     f"IWC of insitu cirrus {insitu_cirrus_idx} : {np.nanmean(self.data['IWC'][current_insitu_cirrus_mask])}"
                    # )
                    # print(
                    #     f"IWC of anvil cirrus {anvil_cirrus_idx} : {np.nanmean(self.data['IWC'][self.extended_anvil_cirrus_mask
                    #     == iteration + 1])}"
                    # )

                    # data = self.data.copy()
                    # data["cld_frac"] = current_insitu_cirrus_mask
                    # processor.filter_and_plot_data_aux_simple_1fig(
                    #     data,
                    #     start_time_idx_manual=start_time_idx,
                    #     time_range_custum_manual=time_range_custum,
                    #     subplot_2_title=f"Current insitu cirrus {insitu_cirrus_idx}",
                    #     subplot_2_clable="Cloud Cluster ID",
                    # )

                    self.extended_anvil_cirrus_mask[
                        current_insitu_cirrus_mask
                    ] = (iteration + 2)
                    self.anvil_cirrus_mask[current_insitu_cirrus_mask] = 1
                    self.aux_anvil_cirrus_mask[current_insitu_cirrus_mask] = (
                        anvil_cirrus_idx
                    )
                    self.insitu_cirrus_mask[current_insitu_cirrus_mask] = 0
                    self.aux_insitu_cirrus_mask[current_insitu_cirrus_mask] = 0

            # data = self.data.copy()
            # data["cld_frac"] = (
            #     self.extended_anvil_cirrus_mask == iteration + 2
            # )
            # processor.filter_and_plot_data_aux_simple_1fig(
            #     data,
            #     start_time_idx_manual=start_time_idx,
            #     time_range_custum_manual=time_range_custum,
            #     subplot_2_title=f"Current insitu cirrus {insitu_cirrus_idx}",
            #     subplot_2_clable="Cloud Cluster ID",
            # )

    def identify_insitu_cirrus_extend_cld(self):
        """
        Identifies and extends the in-situ cirrus cloud layer based on the given data.

        This method processes the cloud cluster data and identifies the in-situ cirrus cloud layer
        by analyzing the cloud layer top, cloud layer base, and temperature data. It then extends
        the identified in-situ cirrus cloud layer by assigning unique indices to each cloud cluster.

        Returns:
            None
        """
        temperature = self.data["temperature"]
        temperature[temperature == -999] = np.nan

        Parallel(n_jobs=-1, backend="threading")(
            delayed(self._process_insitu_cirrus_cluster)(
                idx,
                self.data["cld_layer_top"],
                self.data["cld_layer_base"],
                temperature,
            )
            for idx in range(1, self.insitu_cirrus_idx + 1)
        )

    def _process_insitu_cirrus_cluster(
        self,
        insitu_cluster_idx,
        cloud_layer_top,
        cloud_layer_base,
        temperature,
    ):
        """
        Process a cloud cluster and determine if it meets certain criteria.

        Args:
            cloud_cluster_idx (int): The index of the cloud cluster.
            cloud_layer_top (ndarray): Array containing the top heights of cloud layers.
            cloud_layer_base (ndarray): Array containing the base heights of cloud layers.
            temperature (ndarray): Array containing temperature values.
            cirrus_insitu_idx (int): The index of the cirrus insitu.

        Returns:
            tuple or None: A tuple containing the current cloud cluster mask, cirrus insitu index,
            and a boolean indicating if the cluster meets the criteria. If the cluster does not meet the criteria,
            None is returned.
        """

        current_insitu_cirrus_cluster_mask = (
            self.aux_insitu_cirrus_mask == insitu_cluster_idx
        )

        if np.any(current_insitu_cirrus_cluster_mask):
            profiles_in_cluster = np.any(
                current_insitu_cirrus_cluster_mask, axis=1
            )

            multi_layer_clouds = np.any(
                self.filtered_cloud_type[profiles_in_cluster, 1] != 0
            )

            if (
                # Temperature threshold for cirrus clouds: 233 K
                np.max(temperature[current_insitu_cirrus_cluster_mask]) < 236.15
                # Must contain at least 14 valid profiles
                and np.sum(profiles_in_cluster) > 14
            ):
                if not multi_layer_clouds:
                    return
                else:
                    valid_profile_count = 0
                    cluster_isolated = True

                    for profile_idx in np.where(profiles_in_cluster)[0]:
                        profile_cloud_layers = self.filtered_cloud_type[
                            profile_idx, :
                        ]
                        cirrus_indices = np.where(profile_cloud_layers == 1)[0]

                        if len(cirrus_indices) == 1:
                            if cirrus_indices[0] == 0:
                                valid_profile_count += 1
                                continue

                            if cirrus_indices[0] - 1 < 0:
                                self.insitu_cirrus_mask[
                                    current_insitu_cirrus_cluster_mask
                                ] = 0
                                self.aux_insitu_cirrus_mask[
                                    current_insitu_cirrus_cluster_mask
                                ] = 0
                                cluster_isolated = False
                                break

                            cirrus_top = cloud_layer_top[
                                profile_idx, cirrus_indices[0]
                            ]
                            cirrus_base = cloud_layer_base[
                                profile_idx, cirrus_indices[0]
                            ]
                            other_layers_top = cloud_layer_top[
                                profile_idx, cirrus_indices[0] - 1
                            ]
                            other_layers_base = cloud_layer_base[
                                profile_idx, cirrus_indices[0] - 1
                            ]

                            min_distance = min(
                                cirrus_top - other_layers_base,
                                cirrus_base - other_layers_top,
                            )

                            if min_distance >= 4:
                                valid_profile_count += 1
                            else:
                                self.insitu_cirrus_mask[
                                    current_insitu_cirrus_cluster_mask
                                ] = 0
                                self.aux_insitu_cirrus_mask[
                                    current_insitu_cirrus_cluster_mask
                                ] = 0
                                cluster_isolated = False
                                break

                        if len(cirrus_indices) > 1:

                            if cirrus_indices[0] == 0:
                                valid_profile_count += 1
                                continue

                            cirrus_top = cloud_layer_top[
                                profile_idx, cirrus_indices[0]
                            ]
                            cirrus_base = cloud_layer_base[
                                profile_idx, cirrus_indices[0]
                            ]
                            next_layers_top = cloud_layer_top[
                                profile_idx, cirrus_indices[0] - 1
                            ]
                            next_layers_base = cloud_layer_base[
                                profile_idx, cirrus_indices[0] - 1
                            ]

                            min_distance = min(
                                cirrus_top - next_layers_base,
                                cirrus_base - next_layers_top,
                            )

                            if min_distance >= 4:
                                valid_profile_count += 1
                            else:
                                self.insitu_cirrus_mask[
                                    current_insitu_cirrus_cluster_mask
                                ] = 0
                                self.aux_insitu_cirrus_mask[
                                    current_insitu_cirrus_cluster_mask
                                ] = 0
                                cluster_isolated = False
                                break

                    if cluster_isolated and valid_profile_count > 13:
                        return

            else:
                self.insitu_cirrus_mask[current_insitu_cirrus_cluster_mask] = 0
                self.aux_insitu_cirrus_mask[
                    current_insitu_cirrus_cluster_mask
                ] = 0

        else:
            return

    def main_process(
        self,
        file_index=0,
        dataset_names=["CLDCLASS", "ICE", "GEOPROF", "ECMWF_AUX"],
        iterations=7,
        start_time_idx=None,
        time_range_custum=None,
    ):
        """
        Performs the main processing steps for filtering anvil and insitu cirrus clouds.

        Args:
            file_index (int): The index of the file to process. Defaults to 0.

        Returns:
            data: The processed data.
        """
        self.read_data(file_index=file_index, dataset_names=dataset_names)
        self.create_aux_cld_data()
        self.apply_connected_component_labeling()
        identify_anvil_cirrus, identify_insitu_cirrus = (
            self.filter_cloud_clusters_connected_to_cirrus_or_DC()
        )

        if identify_anvil_cirrus:
            self.iterative_expand_anvil_cirrus(
                iterations=iterations,
                start_time_idx=start_time_idx,
                time_range_custum=time_range_custum,
            )
        if identify_insitu_cirrus:
            self.identify_insitu_cirrus_extend_cld()

        return (
            self.data,
            self.insitu_cirrus_mask,
            self.anvil_cirrus_mask,
            # DCS mask
            self.aux_cld_type == 8,
        )

    def filter_and_plot_data_aux_simple(
        self,
        overall_data,
        start_time_idx_manual,
        time_range_custum_manual,
        subplot_2_title,
        subplot_2_clable,
        custom_color_flag=True,
        custom_color="#e4e4fc",
        custom_cmap="tab20",
        vmin=None,
        vmax=None,
    ):
        """
        Filter and plot CloudSat data.

        Args:
            overall_data (dict): Dictionary containing all CloudSat data.
            time_range_custum (int): Custom time range for filtering data.

        Returns:
            None
        """
        # ----------------------------------------------------------------
        # set the time range
        # Customizable start time (example: start from the first timestamp in the dataset)
        start_time_custom = start_time_idx_manual
        time_range_custum = time_range_custum_manual
        time_range_end = start_time_custom + time_range_custum

        # set hgt
        hgt = self.data["hgt"]

        # Data names to filter
        data_to_filter = [
            "cld_frac",
            "cld_layer_base",
            "cld_layer_top",
            "cld_layer_type",
            "lon",
            "lat",
            "elv",
        ]

        # Initialize a dictionary to store filtered data
        filtered_data = {}

        # Filter the data using the custom start time
        _, time_filtered = filter_by_time_range(
            overall_data["cld_frac"],
            overall_data["time"],
            start_time_custom,
            time_range_end,
        )

        # Apply the filter to each data set
        for data_name in data_to_filter:
            filtered_data[data_name], _ = filter_by_time_range(
                overall_data[data_name],
                overall_data["time"],
                start_time_custom,
                time_range_end,
            )

        # Unpack the filtered data
        cf_time_filtered = filtered_data["cld_frac"]
        (
            cld_layer_base_time_filtered,
            cld_layer_top_time_filtered,
            cld_layer_type_time_filtered,
        ) = (
            filtered_data["cld_layer_base"],
            filtered_data["cld_layer_top"],
            filtered_data["cld_layer_type"],
        )
        _, _, elv = (
            filtered_data["lon"],
            filtered_data["lat"],
            filtered_data["elv"],
        )

        # ----------------------------------------------------------------
        # Plot the data
        # set font
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]

        fig = plt.figure(figsize=(18, 5), dpi=230)

        # Set up the grid
        gs = GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=[30, 1],
            wspace=0.09,
            hspace=0.5,
        )

        ax2 = fig.add_subplot(gs[1 - 1, 0])
        ax3 = fig.add_subplot(gs[2 - 1, 0])

        # Create additional axes for colorbars to the right of the subplots
        cbar_ax2 = fig.add_subplot(gs[1 - 1, 1])
        cbar_ax3 = fig.add_subplot(gs[2 - 1, 1])

        # Add a new subplot for cloud types
        draw_cloud_profile(
            ax2,
            time_filtered,
            cld_layer_base_time_filtered,
            cld_layer_top_time_filtered,
            cld_layer_type_time_filtered,
            hgt,
            cax=cbar_ax2,
        )

        if custom_color_flag:
            # Create a colormap with white for 0 and your specified color for 1
            custom_cmap = ListedColormap(
                [
                    "white",
                    custom_color,
                ]
            )

            # Define the boundaries and norms for the colormap
            boundaries = [0, 0.5, 1]
            norm = BoundaryNorm(boundaries, custom_cmap.N, clip=True)

            draw_cross_section(
                ax3,
                time_filtered,
                overall_data["hgt"],
                cf_time_filtered,
                colormap=custom_cmap,
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax3,
                vmin=vmin,
                vmax=1,
            )

            # Customize the colorbar
            cbar = colorbar.ColorbarBase(
                cbar_ax3,
                cmap=custom_cmap,
                norm=norm,
                boundaries=boundaries,
            )
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(["Clear Sky", "Cirrus"])

        else:
            custom_cmap = plt.get_cmap(custom_cmap)
            custom_cmap.set_bad(color="white")
            draw_cross_section(
                ax3,
                time_filtered,
                overall_data["hgt"],
                cf_time_filtered,
                colormap=custom_cmap,
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax3,
                vmin=vmin,
                vmax=1,
            )

        # Draw elevation in each subplot
        draw_elevation(
            ax2, time_filtered, elv / 1000
        )  # Convert elevation to km
        ax2.set_title("Cloud Types", fontsize=14)
        draw_elevation(
            ax3, time_filtered, elv / 1000
        )  # Convert elevation to km
        ax3.set_title(subplot_2_title, fontsize=14)
        plt.show()


if __name__ == "__main__":

    pass

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Now we extract the in-situ cirrus clouds and the anvil-cirrus clouds
    # in-situ cirrus is stored in array: cirrus_insitu_mask
    # anvil-cirrus is stored in array: cirrus_anvil_mask
    # (^_^) Now you can do whatever you want with these two arrays
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
