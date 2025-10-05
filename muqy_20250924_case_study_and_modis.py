# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:51:04

import gc
import os
import warnings
from datetime import datetime
import numpy.ma as ma
from pyhdf.SD import SD, SDC  # type: ignore

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_dilation, label
from scipy.interpolate import RectBivariateSpline

from muqy_20240101_2Bgeoprof_reader import Reader
from muqy_20240101_plot_2Bgeoprof_test import draw_cross_section
from muqy_20240102_plot_2Bcldtype_2Bcldfrac_2Cice import (
    draw_cloud_profile,
    draw_cross_section,
    draw_elevation,
    filter_by_time_range,
)
from muqy_20240104_filter_anvil_insitu_cirrus import find_and_count_common_files

# Ignore all warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Initiate Satellite data Processor
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Utilities: parse timestamps and find nearest MODIS MYD35 files
# ----------------------------------------------------------------


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

    def filter_and_plot_data_aux_with_micro(
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
        vmax_iwc=None,
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
            "IWC",
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
        iwc_time_filtered = filtered_data["IWC"]
        (
            cld_layer_base_time_filtered,
            cld_layer_top_time_filtered,
            cld_layer_type_time_filtered,
        ) = (
            filtered_data["cld_layer_base"],
            filtered_data["cld_layer_top"],
            filtered_data["cld_layer_type"],
        )
        _, lat, elv = (
            filtered_data["lon"],
            filtered_data["lat"],
            filtered_data["elv"],
        )

        # ----------------------------------------------------------------
        # Plot the data
        # set font
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]

        fig = plt.figure(figsize=(18, 7), dpi=240)

        # Set up the grid
        gs = GridSpec(
            3,
            2,
            figure=fig,
            width_ratios=[30, 1],
            wspace=0.09,
            hspace=0.5,
        )

        ax2 = fig.add_subplot(gs[1 - 1, 0])
        ax3 = fig.add_subplot(gs[2 - 1, 0])
        ax4 = fig.add_subplot(gs[3 - 1, 0])

        # Create additional axes for colorbars to the right of the subplots
        cbar_ax2 = fig.add_subplot(gs[1 - 1, 1])
        cbar_ax3 = fig.add_subplot(gs[2 - 1, 1])
        cbar_ax4 = fig.add_subplot(gs[3 - 1, 1])

        # Add a new subplot for cloud types
        draw_cloud_profile(
            ax2,
            lat,
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
                lat,
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

            iwc_time_filtered = np.where(
                cf_time_filtered == 1, iwc_time_filtered, np.nan
            )

            draw_cross_section(
                ax4,
                lat,
                overall_data["hgt"],
                iwc_time_filtered,
                colormap=plt.get_cmap("cool"),
                cbar_label="IWC (g/m" + r"$^3$" + ")",
                cbar_orientation="vertical",
                cax=cbar_ax4,
                vmin=0,
                vmax=vmax_iwc,
            )

        else:
            custom_cmap = plt.get_cmap(custom_cmap)
            custom_cmap.set_bad(color="white")
            draw_cross_section(
                ax3,
                lat,
                overall_data["hgt"],
                cf_time_filtered,
                colormap=custom_cmap,
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax3,
                vmin=vmin,
                vmax=1,
            )
            draw_cross_section(
                ax4,
                lat,
                overall_data["hgt"],
                iwc_time_filtered,
                colormap=plt.get_cmap("cool"),
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax3,
                vmin=vmin,
                vmax=1,
            )

        # Draw elevation in each subplot
        draw_elevation(ax2, lat, elv / 1000)  # Convert elevation to km
        # ax2.set_title("Cloud Types", fontsize=14)
        draw_elevation(ax3, lat, elv / 1000)  # Convert elevation to km
        # ax3.set_title(subplot_2_title, fontsize=14)
        plt.show()

    def filter_and_plot_data_aux_with_micro_full(
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
        vmax_iwc=None,
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
            "IWC",
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
        iwc_time_filtered = filtered_data["IWC"]
        (
            cld_layer_base_time_filtered,
            cld_layer_top_time_filtered,
            cld_layer_type_time_filtered,
        ) = (
            filtered_data["cld_layer_base"],
            filtered_data["cld_layer_top"],
            filtered_data["cld_layer_type"],
        )
        _, lat, elv = (
            filtered_data["lon"],
            filtered_data["lat"],
            filtered_data["elv"],
        )

        # ----------------------------------------------------------------
        # Plot the data
        # set font
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]

        fig = plt.figure(figsize=(18, 7), dpi=240)

        # Set up the grid
        gs = GridSpec(
            3,
            2,
            figure=fig,
            width_ratios=[30, 1],
            wspace=0.09,
            hspace=0.5,
        )

        ax2 = fig.add_subplot(gs[1 - 1, 0])
        ax3 = fig.add_subplot(gs[2 - 1, 0])
        ax4 = fig.add_subplot(gs[3 - 1, 0])

        # Create additional axes for colorbars to the right of the subplots
        cbar_ax2 = fig.add_subplot(gs[1 - 1, 1])
        cbar_ax3 = fig.add_subplot(gs[2 - 1, 1])
        cbar_ax4 = fig.add_subplot(gs[3 - 1, 1])

        # Add a new subplot for cloud types
        draw_cloud_profile(
            ax2,
            lat,
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
                lat,
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

            iwc_time_filtered = np.where(
                iwc_time_filtered > 0.00001, iwc_time_filtered, np.nan
            )

            draw_cross_section(
                ax4,
                lat,
                overall_data["hgt"],
                iwc_time_filtered,
                colormap=plt.get_cmap("cool"),
                cbar_label="IWC (g/m" + r"$^3$" + ")",
                cbar_orientation="vertical",
                cax=cbar_ax4,
                vmin=0,
                vmax=vmax_iwc,
                draw_contour=False,
            )

        else:
            custom_cmap = plt.get_cmap(custom_cmap)
            custom_cmap.set_bad(color="white")
            draw_cross_section(
                ax3,
                lat,
                overall_data["hgt"],
                cf_time_filtered,
                colormap=custom_cmap,
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax3,
                vmin=vmin,
                vmax=1,
            )
            draw_cross_section(
                ax4,
                lat,
                overall_data["hgt"],
                iwc_time_filtered,
                colormap=plt.get_cmap("cool"),
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax3,
                vmin=vmin,
                vmax=1,
            )

        # Draw elevation in each subplot
        draw_elevation(ax2, lat, elv / 1000)  # Convert elevation to km
        # ax2.set_title("Cloud Types", fontsize=14)
        draw_elevation(ax3, lat, elv / 1000)  # Convert elevation to km
        # ax3.set_title(subplot_2_title, fontsize=14)
        plt.show()

    def filter_and_plot_data_aux_simple_1fig(
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
            1,
            2,
            figure=fig,
            width_ratios=[30, 1],
            wspace=0.09,
        )

        ax2 = fig.add_subplot(gs[1 - 1, 0])

        # Create additional axes for colorbars to the right of the subplots
        cbar_ax2 = fig.add_subplot(gs[1 - 1, 1])

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
                ax2,
                time_filtered,
                overall_data["hgt"],
                cf_time_filtered,
                colormap=custom_cmap,
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax2,
                vmin=vmin,
                vmax=1,
            )

            # Customize the colorbar
            cbar = colorbar.ColorbarBase(
                cbar_ax2,
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
                ax2,
                time_filtered,
                overall_data["hgt"],
                cf_time_filtered,
                colormap=custom_cmap,
                cbar_label=subplot_2_clable,
                cbar_orientation="vertical",
                cax=cbar_ax2,
                vmin=vmin,
                vmax=1,
            )

        # Draw elevation in each subplot
        draw_elevation(
            ax2, time_filtered, elv / 1000
        )  # Convert elevation to km
        ax2.set_title("Cloud Types", fontsize=14)
        draw_elevation(
            ax2, time_filtered, elv / 1000
        )  # Convert elevation to km
        ax2.set_title(subplot_2_title, fontsize=14)
        plt.show()


def _parse_cloudsat_start_time_from_filename(file_path):
    """Parse CloudSat start time from filename like '2006166031842_...hdf'.

    Returns a timezone-naive UTC datetime or None if parsing fails.
    """
    import os
    import re
    from datetime import datetime, timedelta

    base = os.path.basename(file_path)
    m = re.search(r"(\d{13})", base)
    if not m:
        return None
    s = m.group(1)
    year = int(s[0:4])
    doy = int(s[4:7])
    hh = int(s[7:9])
    mm = int(s[9:11])
    ss = int(s[11:13])
    return datetime(year, 1, 1) + timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss
    )


def _parse_myd35_time_from_filename(file_path):
    """Parse MODIS MYD35_L2 time from filename like 'MYD35_L2.A2006181.2350....hdf'.

    Returns a timezone-naive UTC datetime or None if parsing fails.
    """
    import os
    import re
    from datetime import datetime, timedelta

    base = os.path.basename(file_path)
    m = re.match(r"^MYD35_L2\.A(\d{7})\.(\d{4})", base)
    if not m:
        return None
    yddd = m.group(1)
    hhmm = m.group(2)
    year = int(yddd[0:4])
    doy = int(yddd[4:7])
    hh = int(hhmm[0:2])
    mm = int(hhmm[2:4])
    return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm)


def find_nearest_myd35_files(cloudsat_file, myd35_root, k=3):
    """Find k MYD35 files nearest in time to the given CloudSat granule.

    - cloudsat_file: absolute path to a CloudSat 2B_CLDCLASS_LIDAR file
    - myd35_root: root folder containing many MYD35 L2 files (searched recursively)
    - k: number of nearest files to return

    Returns a list of tuples: (abs_time_diff_seconds, myd35_datetime, myd35_path)
    """
    import os

    target_dt = _parse_cloudsat_start_time_from_filename(cloudsat_file)
    if target_dt is None:
        return []

    candidates = []
    for root, _, files in os.walk(myd35_root):
        for name in files:
            if not name.endswith(".hdf"):
                continue
            if not name.startswith("MYD35_L2.A"):
                continue
            fpath = os.path.join(root, name)
            dt = _parse_myd35_time_from_filename(fpath)
            if dt is None:
                continue
            diff = abs((dt - target_dt).total_seconds())
            candidates.append((diff, dt, fpath))

    candidates.sort(key=lambda x: x[0])
    return candidates[:k]


def get_nearest_myd35_file(cloudsat_file, myd35_root):
    """Return the single nearest MYD35 file path (or None)."""
    nearest = find_nearest_myd35_files(cloudsat_file, myd35_root, k=3)
    if nearest:
        return nearest[0][2]
    return None


def inspect_modis_file(hdf_path, max_print_sds=50, compute_stats=True):
    """Print structure/content of a MODIS MYD35 HDF file.

    Parameters
    ----------
    hdf_path : str
        Path to MYD35_L2 *.hdf file.
    max_print_sds : int
        Limit number of SDS entries to print (safety for huge files).
    compute_stats : bool
        Whether to compute min/max/mean for numeric SDS.
    """
    from pyhdf.SD import SD, SDC  # type: ignore

    print(f"\n=== Inspect MODIS HDF: {hdf_path} ===")
    try:
        sd = SD(hdf_path, SDC.READ)
    except Exception as e:
        print(f"打开文件失败: {e}")
        return

    # Global attributes
    print("-- Global Attributes --")
    try:
        gattrs = sd.attributes()
        for k, v in gattrs.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"读取全局属性失败: {e}")

    # Scientific Data Sets (SDS)
    print("\n-- SDS Datasets (name: shape | type | attributes) --")
    try:
        sds_dict = sd.datasets()
    except Exception as e:
        print(f"列出 SDS 失败: {e}")
        sds_dict = {}

    count = 0
    for name, meta in sds_dict.items():
        if count >= max_print_sds:
            print(f"  ... (已达到打印上限 {max_print_sds})")
            break
        idx, rank, dims, dtype, nattrs = meta
        print(f"  {name}: {dims} | {dtype} | attrs={nattrs}")
        try:
            sds_obj = sd.select(idx)
            attrs = sds_obj.attributes()
            scale = attrs.get('scale_factor') or attrs.get('scale')
            offset = attrs.get('add_offset') or attrs.get('offset')
            uf = attrs.get('units') or attrs.get('unit')
            fill = (
                attrs.get('_FillValue')
                or attrs.get('fillvalue')
                or attrs.get('missing_value')
            )
            if uf or scale or offset or fill is not None:
                print(
                    f"     -> units={uf} scale={scale} offset={offset} fill={fill}"
                )
            if compute_stats and rank > 0 and len(dims) > 0:
                import numpy as np

                arr = (
                    sds_obj.get()
                )  # careful large arrays; MYD35 typical dims manageable
                # mask fill if provided
                if fill is not None:
                    arr = np.where(arr == fill, np.nan, arr)
                finite = np.isfinite(arr)
                if np.any(finite):
                    sample = arr[finite]
                    print(
                        f"     -> stats: min={np.nanmin(sample):.3f} max={np.nanmax(sample):.3f} mean={np.nanmean(sample):.3f}"
                    )
        except Exception as e:
            print(f"     -> 读取/统计失败: {e}")
        count += 1

    # Vdata (简略尝试)
    print("\n(未实现 Vdata 遍历，如需要可后续扩展)")
    try:
        sd.end()
    except Exception:
        pass


def decode_cloud_mask(cloud_mask_raw):
    """Decode MODIS MYD35 Cloud_Mask first byte (and second, third, fourth, fifth, sixth) into meaningful layers.

    cloud_mask_raw: expected shape (Byte_Segment, Along, Across). Usually Byte_Segment >= 6 for needed flags.

    Returns dict with keys:
      - cloud_mask_determined (bool)
      - unobstructed_fov_quality (0-3)
      - day_flag (0 night /1 day)
      - sunglint (0 yes /1 no)
      - snow_ice_background (0 yes /1 no)
      - land_water_type (0-3) 00 water,01 coastal,10 desert,11 land
      - thin_cirrus_solar (0 yes /1 no)
      - thin_cirrus_ir (0 yes /1 no)
      - high_cloud_co2 (0 yes /1 no)
      - high_cloud_67 (0 yes /1 no)
      - high_cloud_138 (0 yes /1 no)
      - high_cloud_39_12 (0 yes /1 no)
      - cloud_ir_temp_diff (0 yes /1 no)
      - etc... (can be extended)
    """
    # Ensure numpy array uint8
    arr = np.asarray(cloud_mask_raw)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Byte 0 (index 0)
    b0 = arr[0]
    # bit positions (7 MSB ... 0 LSB) -> use shifts
    # Cloud Mask Flag (bit0) 0 not determined /1 determined
    cloud_mask_determined = (b0 & 0b00000001) == 1
    # Unobstructed FOV Quality Flag bits1-2 (2,1 in doc) -> extract bits1-2
    unobstructed_fov_quality = (b0 & 0b00000110) >> 1  # 0..3 mapping doc
    day_flag = (b0 & 0b00001000) >> 3
    sunglint = (b0 & 0b00010000) >> 4
    snow_ice_background = (b0 & 0b00100000) >> 5
    land_water_type = (b0 & 0b11000000) >> 6

    # Byte 1 (index 1) 1-km flags
    if arr.shape[0] > 1:
        b1 = arr[1]
    else:
        b1 = np.zeros_like(b0)
    non_cloud_obstruction = (b1 & 0b00000001) == 1
    thin_cirrus_solar = (b1 & 0b00000010) >> 1  # 0 yes /1 no
    snow_cover_map = (b1 & 0b00000100) >> 2
    thin_cirrus_ir = (b1 & 0b00001000) >> 3
    cloud_adjacency = (b1 & 0b00010000) >> 4
    cloud_flag_ir_threshold = (b1 & 0b00100000) >> 5
    high_cloud_co2 = (b1 & 0b01000000) >> 6
    high_cloud_67 = (b1 & 0b10000000) >> 7

    # Byte 2 (index 2)
    if arr.shape[0] > 2:
        b2 = arr[2]
    else:
        b2 = np.zeros_like(b0)
    high_cloud_138 = (b2 & 0b00000001) >> 0
    high_cloud_39_12 = (b2 & 0b00000010) >> 1
    cloud_flag_ir_temp_diff = (b2 & 0b00000100) >> 2
    cloud_flag_39_11 = (b2 & 0b00001000) >> 3
    cloud_flag_vis_refl = (b2 & 0b00010000) >> 4
    cloud_flag_visnir_ratio = (b2 & 0b00100000) >> 5
    cloud_flag_ndvi_restoral = (b2 & 0b01000000) >> 6
    cloud_flag_night_land_polar_73_11 = (b2 & 0b10000000) >> 7

    # Byte 3 (index 3)
    if arr.shape[0] > 3:
        b3 = arr[3]
    else:
        b3 = np.zeros_like(b0)
    cloud_flag_ocean_86_11 = (b3 & 0b00000001) >> 0
    cloud_flag_clear_restoral_spatial = (b3 & 0b00000010) >> 1
    cloud_flag_clear_restoral_polar = (b3 & 0b00000100) >> 2
    cloud_flag_surface_temp = (b3 & 0b00001000) >> 3
    suspended_dust_flag = (b3 & 0b00010000) >> 4
    cloud_flag_night_ocean_86_73 = (b3 & 0b00100000) >> 5
    cloud_flag_night_ocean_11_spatial_var = (b3 & 0b01000000) >> 6
    cloud_flag_night_ocean_low_emissivity = (b3 & 0b10000000) >> 7

    decoded = {
        'cloud_mask_determined': cloud_mask_determined,
        'unobstructed_fov_quality': unobstructed_fov_quality,
        'day_flag': day_flag,
        'sunglint': sunglint,
        'snow_ice_background': snow_ice_background,
        'land_water_type': land_water_type,
        'non_cloud_obstruction': non_cloud_obstruction,
        'thin_cirrus_solar': thin_cirrus_solar,
        'snow_cover_map': snow_cover_map,
        'thin_cirrus_ir': thin_cirrus_ir,
        'cloud_adjacency': cloud_adjacency,
        'cloud_flag_ir_threshold': cloud_flag_ir_threshold,
        'high_cloud_co2': high_cloud_co2,
        'high_cloud_67': high_cloud_67,
        'high_cloud_138': high_cloud_138,
        'high_cloud_39_12': high_cloud_39_12,
        'cloud_flag_ir_temp_diff': cloud_flag_ir_temp_diff,
        'cloud_flag_39_11': cloud_flag_39_11,
        'cloud_flag_vis_refl': cloud_flag_vis_refl,
        'cloud_flag_visnir_ratio': cloud_flag_visnir_ratio,
        'cloud_flag_ndvi_restoral': cloud_flag_ndvi_restoral,
        'cloud_flag_night_land_polar_73_11': cloud_flag_night_land_polar_73_11,
        'cloud_flag_ocean_86_11': cloud_flag_ocean_86_11,
        'cloud_flag_clear_restoral_spatial': cloud_flag_clear_restoral_spatial,
        'cloud_flag_clear_restoral_polar': cloud_flag_clear_restoral_polar,
        'cloud_flag_surface_temp': cloud_flag_surface_temp,
        'suspended_dust_flag': suspended_dust_flag,
        'cloud_flag_night_ocean_86_73': cloud_flag_night_ocean_86_73,
        'cloud_flag_night_ocean_11_spatial_var': cloud_flag_night_ocean_11_spatial_var,
        'cloud_flag_night_ocean_low_emissivity': cloud_flag_night_ocean_low_emissivity,
    }
    return decoded


def print_nearest_myd35_for_cloudsat_files(cloudsat_files, myd35_root, k=3):
    """Pretty-print the k-nearest MYD35 files for each CloudSat file."""
    import os

    for cs_file in cloudsat_files:
        cs_dt = _parse_cloudsat_start_time_from_filename(cs_file)
        results = find_nearest_myd35_files(cs_file, myd35_root, k=k)
        print("\n============================")
        print(f"CloudSat file: {cs_file}")
        print(f"CloudSat start (UTC): {cs_dt}")
        if not results:
            print("No MYD35 candidates found.")
            continue
        for i, (diff_s, dt, path) in enumerate(results, start=1):
            mins = int(round(diff_s / 60.0))
            print(
                f"  {i}. {os.path.basename(path)} | {dt} UTC | Δt ≈ {mins} min"
            )


def _safe_get(sd, name):
    """安全地从HDF对象中读取数据集。"""
    try:
        s = sd.select(name)
        return s.get()
    except Exception:
        print(f"Warning: Unable to read dataset '{name}'")
        return None


def process_modis_data(modis_filepath, downsample_factor=3):
    """
    处理单个MODIS文件：读取、降采样、插值经纬度。
    返回与云掩码匹配的经纬度网格和云掩码本身。
    """
    print(f"--- 正在处理 MODIS 文件: {modis_filepath.split('/')[-1]}")
    sd = None
    try:
        sd = SD(modis_filepath, SDC.READ)
        cloud_mask_raw = _safe_get(sd, 'Cloud_Mask')
        lat_5km = _safe_get(sd, 'Latitude')
        lon_5km = _safe_get(sd, 'Longitude')

        if cloud_mask_raw is None or lat_5km is None or lon_5km is None:
            raise ValueError(
                "文件缺少关键数据 (Cloud_Mask, Latitude, or Longitude)"
            )

        # 解码并降采样云掩码
        decoded_data = decode_cloud_mask(cloud_mask_raw)
        is_cloudy_mask = decoded_data['unobstructed_fov_quality'] == 0
        mask_resampled = is_cloudy_mask[
            ::downsample_factor, ::downsample_factor
        ]

        target_shape = mask_resampled.shape

        # 插值经纬度以匹配降采样后的云掩码
        source_shape = lat_5km.shape
        y_source = np.arange(source_shape[0])
        x_source = np.arange(source_shape[1])

        interp_lat = RectBivariateSpline(y_source, x_source, lat_5km)
        interp_lon = RectBivariateSpline(y_source, x_source, lon_5km)

        y_target = np.linspace(0, source_shape[0] - 1, target_shape[0])
        x_target = np.linspace(0, source_shape[1] - 1, target_shape[1])

        lat_resampled = interp_lat(y_target, x_target)
        lon_resampled = interp_lon(y_target, x_target)

        print("--- MODIS 数据处理完成。")
        return lon_resampled, lat_resampled, mask_resampled

    except Exception as e:
        print(f"处理 MODIS 文件时出错: {e}")
        return None, None, None
    finally:
        if sd:
            sd.end()


def process_cloudsat_data(params):
    """处理单个CloudSat案例，返回完整的轨道经纬度和时间。"""
    print("--- 正在处理 CloudSat 数据...")
    processor = CloudSatProcessor_Ver2(
        common_files,
        structure_0=params["structure_0"],
        structure_1=params["structure_1"],
    )
    data, _, _, _ = processor.main_process(
        file_index=params["file_index"],
        dataset_names=[
            "CLDCLASS",
            "ICE",
            "GEOPROF",
            "ECMWF_AUX",
        ],  # 仅读取轨道数据以加速
        iterations=1,
        start_time_idx=0,  # 读取完整轨道
        time_range_custum=99999,  # 读取完整轨道
    )

    print("--- CloudSat 数据处理完成。")
    # 确保返回的是1D数组
    return data['lat'].flatten(), data['lon'].flatten(), data['time'].flatten()


def visualize_cloud_mask(
    lon_2d,
    lat_2d,
    mask_2d,
    # --- 轨道参数 ---
    full_track_lon=None,
    full_track_lat=None,
    full_track_color='gray',
    full_track_linewidth=1.5,
    full_track_alpha=0.8,
    full_track_label='Full CloudSat Track',
    # --- 新增：高亮轨道参数 ---
    highlight_lon=None,
    highlight_lat=None,
    highlight_color='red',
    highlight_linewidth=3,
    highlight_label='Region of Interest',
    # --- 其他参数保持不变 ---
    projection=ccrs.PlateCarree(),
    # extent 为 (west_lon, east_lon, south_lat, north_lat) -> (西经, 东经, 南纬, 北纬)，单位：度
    # 例：extent=(-130, -60, 10, 50) 表示北美局部；经度范围建议在 [-180, 180] 或 [0, 360] 之一保持一致
    extent=None,
    cloud_color='white',
    alpha=0.7,
    figsize=(8, 10),
    dpi=280,
    # --- 新增：extent 参数控制 ---
    validate_extent=True,  # 若为 True，将检查顺序与范围并给出清晰报错
    extent_crs=ccrs.PlateCarree(),  # extent 数字对应的坐标系（通常为地理经纬度 PlateCarree）
    fallback_on_invalid_extent=True,  # 当 extent 非法时，是否回退为不设置 extent（继续绘图）
):
    """
    使用 Cartopy 可视化 MODIS 云掩码数据（带底图增强版）。
    通过蒙版数组确保无云区域完全透明。

    extent 参数说明（严格的顺序与含义）：
    - extent=(west_lon, east_lon, south_lat, north_lat)
      -> west_lon  西经（左边界，经度）
      -> east_lon  东经（右边界，经度）
      -> south_lat 南纬（下边界，纬度）
      -> north_lat 北纬（上边界，纬度）
    单位：度。常见经度范围为 [-180, 180] 或 [0, 360]，两者请保持一致；
    维度范围为 [-90, 90]。要求 west_lon < east_lon 且 south_lat < north_lat。

    性能诊断提示：
    - 主要耗时通常在 pcolormesh 的重投影/网格变换上（2D lon/lat + transform）。
    - 底图绘制（coastlines、borders、stock_img）和网格线标注也会显著增加绘制时间。
    - 大尺寸画布/高 DPI、全局范围、图例布局都会增加渲染开销。
    - 如需快速定位瓶颈，可临时注释掉 pcolormesh 或底图要素，逐项恢复观察耗时。
    """
    # 设置全局字体为times new roman
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    def _apply_extent(ax, ext, crs, do_validate=True):
        """验证并应用 extent。"""
        try:
            if not isinstance(ext, (list, tuple)) or len(ext) != 4:
                raise ValueError(
                    "extent 必须是长度为 4 的序列：(west_lon, east_lon, south_lat, north_lat)"
                )
            west, east, south, north = map(float, ext)

            if do_validate:
                # 顺序校验
                if not (west < east):
                    raise ValueError(
                        f"extent 经度顺序错误：要求 west_lon < east_lon，当前为 ({west}, {east})"
                    )
                if not (south < north):
                    raise ValueError(
                        f"extent 纬度顺序错误：要求 south_lat < north_lat，当前为 ({south}, {north})"
                    )
                # 合理范围校验（仅在 PlateCarree 下检查）
                if isinstance(crs, ccrs.PlateCarree):
                    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
                        raise ValueError(
                            f"extent 纬度超范围：应在 [-90, 90] 内，当前为 ({south}, {north})"
                        )
                    # 经度常见范围，允许 0..360 或 -180..180
                    if not (
                        (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0)
                        or (0.0 <= west <= 360.0 and 0.0 <= east <= 360.0)
                    ):
                        raise ValueError(
                            f"extent 经度超常见范围：建议使用 [-180,180] 或 [0,360]，当前为 ({west}, {east})"
                        )

            ax.set_extent([west, east, south, north], crs=crs)
            return True
        except Exception as e:
            print(f"[WARN] 无法应用 extent：{e}")
            return False

    # --- 核心改动：创建蒙版数组 ---
    mask_to_plot = ma.masked_where(mask_2d == False, mask_2d)

    # --- 简化 Colormap ---
    custom_cmap = mcolors.ListedColormap([cloud_color])

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # 在绘制前优先设置 extent（若用户提供），避免底图全局渲染
    if extent is not None:
        ok = _apply_extent(ax, extent, extent_crs, do_validate=validate_extent)
        if not ok and not fallback_on_invalid_extent:
            # 直接抛错，停止绘图
            raise ValueError("提供的 extent 非法，且未允许回退。")

    mesh = ax.pcolormesh(
        lon_2d,
        lat_2d,
        mask_to_plot,  # 使用蒙版数组
        cmap=custom_cmap,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        shading='nearest',
        rasterized=True,
    )

    # 绘制轨道
    if full_track_lon is not None and full_track_lat is not None:
        ax.plot(
            full_track_lon,
            full_track_lat,
            color=full_track_color,
            linewidth=full_track_linewidth,
            alpha=full_track_alpha,
            label=full_track_label,
            transform=ccrs.PlateCarree(),
        )

    if highlight_lon is not None and highlight_lat is not None:
        ax.plot(
            highlight_lon,
            highlight_lat,
            color=highlight_color,
            linewidth=highlight_linewidth,
            label=highlight_label,
            transform=ccrs.PlateCarree(),
        )

    ax.legend(loc='best')

    # 地理元素与底图
    ax.coastlines(color='yellow', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='yellow')
    ax.stock_img()

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color='white',
        alpha=0.6,
        linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False

    plt.show()


def visualize_track(
    # --- 轨道参数 ---
    full_track_lon=None,
    full_track_lat=None,
    full_track_color='gray',
    full_track_linewidth=1.5,
    full_track_alpha=0.8,
    full_track_label='Full CloudSat Track',
    # --- 新增：高亮轨道参数 ---
    highlight_lon=None,
    highlight_lat=None,
    highlight_color='red',
    highlight_linewidth=3,
    highlight_label='Region of Interest',
    # --- 其他参数保持不变 ---
    projection=ccrs.PlateCarree(),
    extent=None,
    basemap='satellite',
    title='MODIS Cloud Mask with Highlighted CloudSat Track',
    cloud_color='white',
    alpha=0.7,
    figsize=(12, 10),
    dpi=240,
):
    """
    使用 Cartopy 可视化 MODIS 云掩码数据（带底图增强版）。
    通过蒙版数组确保无云区域完全透明。
    """
    import numpy.ma as ma

    cache_dir = "py_cache_data"
    os.makedirs(cache_dir, exist_ok=True)
    ctx.set_cache_dir(cache_dir)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.set_global()

    # 添加底图
    ax.add_feature(
        cfeature.LAND.with_scale('10m'), facecolor='peru'
    )  # 沙色陆地
    ax.add_feature(
        cfeature.OCEAN.with_scale('10m'), facecolor='#d0eaf7'
    )  # 浅蓝海洋

    # --- 核心新增逻辑：绘制轨道 ---
    if full_track_lon is not None and full_track_lat is not None:
        ax.plot(
            full_track_lon,
            full_track_lat,
            color=full_track_color,
            linewidth=full_track_linewidth,
            alpha=full_track_alpha,
            label=full_track_label,
            transform=ccrs.PlateCarree(),
        )

    # 2. 在上方绘制高亮的轨道片段
    if highlight_lon is not None and highlight_lat is not None:
        ax.plot(
            highlight_lon,
            highlight_lat,
            color=highlight_color,
            linewidth=highlight_linewidth,
            label=highlight_label,
            transform=ccrs.PlateCarree(),
        )

    ax.legend(loc='best')  # 自动选择最佳位置显示图例

    # 添加地理元素
    ax.coastlines(color='yellow', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='yellow')

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color='white',
        alpha=0.6,
        linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=16)

    plt.show()


if __name__ == "__main__":

    # Windows paths
    base_paths = {
        "CLDCLASS": r"P:\CLOUDSAT_DATA\2B_CLDCLASS_LIDAR",
        "ICE": r"P:\CLOUDSAT_DATA\2C_ICE",
        "GEOPROF": r"P:\CLOUDSAT_DATA\2B_GEOPROF_LIDAR",
        "ECMWF_AUX": r"G:\Data\CLOUDSAT_DATA\ECMWF_AUX",
        "MODIS_AUX": r"D:\MODIS_AUX",
        # Add other datasets if necessary
    }

    # find and count common files
    common_files, common_file_count = find_and_count_common_files(base_paths)

    # ----------------------------------------------------------------
    # Locate nearest MYD35_L2 files for the two specified CloudSat granules
    # ----------------------------------------------------------------
    # myd35_root = "/RAID01/data/Satellite_Data/MODIS_MYD35"
    myd35_root = r"E:\MODIS_MYD35"

    cloudsat_targets = [
        r"P:\CLOUDSAT_DATA\2B_CLDCLASS_LIDAR\2006\166\2006166031842_00696_CS_2B-CLDCLASS-LIDAR_GRANULE_P1_R05_E00_F00.hdf",
        r"P:\CLOUDSAT_DATA\2B_CLDCLASS_LIDAR\2006\179\2006179110219_00890_CS_2B-CLDCLASS-LIDAR_GRANULE_P1_R05_E00_F00.hdf",
        r"P:\CLOUDSAT_DATA\2B_CLDCLASS_LIDAR\2006\172\2006172105606_00788_CS_2B-CLDCLASS-LIDAR_GRANULE_P1_R05_E00_F00.hdf",
    ]

    print_nearest_myd35_for_cloudsat_files(cloudsat_targets, myd35_root, k=4)

    # ----------------------------------------------------------------
    # 读取MODIS匹配的数据，并进行降采样处理以加速后续操作
    # ----------------------------------------------------------------
    # region

    CASE_DEFINITIONS = [
        {
            "case_name": "Article Figure 1",
            "cloudsat_target_idx": 0,
            "file_index": 30,
            "structure_0": np.ones((3, 1)),
            "structure_1": np.ones((1, 3)),
            "highlight_start_idx": 3570,
            "highlight_range": 200,
            # --- 新增：可选的自定义参数 ---
            "downsample_factor": 3,  # 为此案例指定特定的降采样因子
            "highlight_color": "red",  # 自定义高亮轨道的颜色
            "title_override": "Custom Title for Figure 1 Case",  # 完全覆盖默认标题
            # "modis_override_path": "E:\\SOME_OTHER_MODIS_FILE.hdf", # 可选：手动指定MODIS文件路径
        },
        {
            "case_name": "Article Figure 2",
            "cloudsat_target_idx": 1,
            "file_index": 169,
            "structure_0": np.ones((3, 1)),
            "structure_1": np.ones((1, 3)),
            "highlight_start_idx": 2950,
            "highlight_range": 150,
            # --- 新增：可选的自定义参数 ---
            # extent 为 (west_lon, east_lon, south_lat, north_lat) -> (西经, 东经, 南纬, 北纬)，单位：度
            # 例：extent=(-130, -60, 10, 50) 表示北美局部；经度范围建议在 [-180, 180] 或 [0, 360] 之一保持一致
            'extent': (-165, -130, -35, -9),  # 北美局部范围
            "downsample_factor": 2,  # 为此案例指定特定的降采样因子
            "highlight_color": "red",  # 自定义高亮轨道的颜色
            "title_override": "Custom Title for Figure 1 Case",  # 完全覆盖默认标题
            "modis_override_path": "E:\\MODIS_MYD35\\MYD35_L2.A2006179.1105.061.2018023011243.hdf",  # 可选：手动指定MODIS文件路径
        },
        {
            "case_name": "Article Figure 3",
            "cloudsat_target_idx": 2,
            "file_index": 69,
            "structure_0": np.ones((3, 1)),
            "structure_1": np.ones((1, 3)),
            "highlight_start_idx": 3650,
            "highlight_range": 470,
            # --- 新增：可选的自定义参数 ---
            # extent 为 (west_lon, east_lon, south_lat, north_lat) -> (西经, 东经, 南纬, 北纬)，单位：度
            # 例：extent=(-130, -60, 10, 50) 表示北美局部；经度范围建议在 [-180, 180] 或 [0, 360] 之一保持一致
            'extent': (-172, -135, -60, -30),  # 北美局部范围
            "downsample_factor": 2,  # 为此案例指定特定的降采样因子
            "highlight_color": "red",  # 自定义高亮轨道的颜色
            "title_override": None,  # 完全覆盖默认标题
            # "modis_override_path": "E:\\MODIS_MYD35\\MYD35_L2.A2006172.1100.061.2018023001820.hdf",  # 可选：手动指定MODIS文件路径
            "modis_override_path": "E:\\MODIS_MYD35\\MYD35_L2.A2006172.1105.061.2018023001849.hdf",  # 可选：手动指定MODIS文件路径
        },
    ]

    def run_and_visualize_case(case_config):
        """
        执行并可视化一个完整的案例。
        现在会从 case_config 中读取更多自定义设置。
        """
        case_name = case_config["case_name"]
        print(f"\n{'='*25}\n   开始运行案例: {case_name}\n{'='*25}")

        # --- 从配置中安全地获取参数 (使用.get()提供默认值) ---
        downsample_factor = case_config.get(
            "downsample_factor", 3
        )  # 默认降采样因子为3
        full_track_color = case_config.get(
            "full_track_color", "gray"
        )  # 默认轨道颜色
        highlight_color = case_config.get(
            "highlight_color", "red"
        )  # 默认高亮颜色

        # --- 步骤 1: 处理MODIS数据 (支持手动覆盖) ---
        modis_override = case_config.get("modis_override_path")

        if modis_override:
            print(f"*** 使用手动指定的MODIS文件: {modis_override}")
            modis_filepath = modis_override
        else:
            print("--- 自动查找匹配的MODIS文件...")
            cs_target = cloudsat_targets[case_config["cloudsat_target_idx"]]
            modis_filepath = get_nearest_myd35_file(cs_target, myd35_root)

        modis_lon, modis_lat, modis_mask = process_modis_data(
            modis_filepath, downsample_factor
        )
        if modis_lon is None:
            print(f"案例 {case_name} 失败：无法处理 MODIS 数据。")
            return

        # --- 步骤 2: 处理CloudSat数据 (逻辑不变) ---
        full_lat, full_lon, full_time = process_cloudsat_data(case_config)
        if full_lon is None:
            print(f"案例 {case_name} 失败：无法处理 CloudSat 数据。")
            return

        # --- 步骤 3: 计算高亮轨道片段 (逻辑不变) ---
        start_idx = case_config['highlight_start_idx']
        end_idx = start_idx + case_config['highlight_range']
        highlight_lon_segment = full_lon[start_idx:end_idx]
        highlight_lat_segment = full_lat[start_idx:end_idx]

        # --- 步骤 4: 调用可视化函数 (使用自定义参数) ---
        print("\n--- 所有数据准备就绪，开始绘图... ---")
        visualize_cloud_mask(
            modis_lon,
            modis_lat,
            modis_mask,
            full_track_lon=full_lon,
            full_track_lat=full_lat,
            full_track_color=full_track_color,  # 传递自定义颜色
            highlight_lon=highlight_lon_segment,
            highlight_lat=highlight_lat_segment,
            highlight_color=highlight_color,  # 传递自定义颜色
            extent=case_config['extent'],
        )
        print(f"--- 案例 {case_name} 执行完毕 ---")
        gc.collect()

    # ----------------------------------------------------------------
    # 案例调试
    # ----------------------------------------------------------------
    # 函数现在直接从config内部读取降采样因子和其他设置
    # 调试案例1
    run_and_visualize_case(CASE_DEFINITIONS[2])

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # @param article case0
    # ----------------------------------------------------------------

    # 现在开始该函数会返回结果，结果为data，用于读取CloudSat轨迹的经纬度数据
    def plot_article_case0(
        file_index=26,
        structure_0=np.ones((10, 1)),
        structure_1=np.ones((1, 3)),
        start_time_idx=0,
        time_range_custum=80,
        vmax_iwc=0.005,
    ):
        """Plot article case 0 figures showing anvil and in-situ cirrus clouds

        Args:
            file_index (int): Index of file to process
            start_time_idx (int): Start time index
            time_range_custum (int): Time range to plot

        Returns:
            None
        """
        print(f"now processing file idx: {file_index}")
        print(f"corrently testing file: {common_files['CLDCLASS'][file_index]}")

        processor = CloudSatProcessor_Ver2(
            common_files,
            structure_0=structure_0,  # vertical extent
            structure_1=structure_1,  # horizontal extent
        )
        data, cirrus_insitu_mask, cirrus_anvil_mask, DCS_mask = (
            processor.main_process(
                file_index=file_index,
                dataset_names=[
                    "CLDCLASS",
                    "ICE",
                    "GEOPROF",
                    "ECMWF_AUX",
                ],
                iterations=25,
                start_time_idx=start_time_idx,
                time_range_custum=time_range_custum,
            )
        )

        # Anvil cirrus
        # cirrus_anvil_mask = np.where(cirrus_anvil_mask == 1, 1, np.nan)
        data["cld_frac"] = cirrus_anvil_mask
        processor.filter_and_plot_data_aux_with_micro(
            data,
            start_time_idx_manual=start_time_idx,
            time_range_custum_manual=time_range_custum,
            subplot_2_title="Cloud Cluster",
            subplot_2_clable="Cloud Cluster ID",
            vmax_iwc=vmax_iwc,
        )
        processor.filter_and_plot_data_aux_with_micro_full(
            data,
            start_time_idx_manual=start_time_idx,
            time_range_custum_manual=time_range_custum,
            subplot_2_title="Cloud Cluster",
            subplot_2_clable="Cloud Cluster ID",
            vmax_iwc=vmax_iwc,
        )

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Figure 1 in the article
    start_time_idx = 3570
    time_range_custum = 200

    plot_article_case0(
        file_index=30,
        structure_0=np.ones((11, 1)),
        structure_1=np.ones((1, 3)),
        start_time_idx=start_time_idx,
        time_range_custum=time_range_custum,
        vmax_iwc=0.03,
    )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Figure 2 in the article
    start_time_idx = 2950
    time_range_custum = 150

    plot_article_case0(
        file_index=169,
        structure_0=np.ones((16, 1)),
        structure_1=np.ones((1, 6)),
        start_time_idx=start_time_idx,
        time_range_custum=time_range_custum,
        vmax_iwc=0.03,
    )

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Might be a good case
    start_time_idx = 3650
    time_range_custum = 470

    plot_article_case0(
        file_index=69,
        structure_0=np.ones((26, 1)),
        structure_1=np.ones((1, 4)),
        start_time_idx=start_time_idx,
        time_range_custum=time_range_custum,
        vmax_iwc=0.03,
    )
