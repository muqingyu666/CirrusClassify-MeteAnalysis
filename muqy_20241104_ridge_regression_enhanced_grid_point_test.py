# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-08-20 08:38
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:47:46

import logging
import warnings
from matplotlib import gridspec
import seaborn as sns

import cartopy.crs as ccrs
import matplotlib as mpl
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Disable all warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging to display info messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

####################################################################################################


def perform_kfold_cv(
    X,
    y,
    n_splits=5,
    alpha=1.0,
    use_grid_search=False,
    param_grid=None,
    cv_strategy=None,
    scoring_metric="r2",
    n_jobs=-1,
    verbosity=0,
):
    """
    Performs K-Fold Cross-Validation with Ridge Regression.

    Args:
        X (ndarray): Predictor data.
        y (ndarray): Target data.
        n_splits (int): Number of splits for KFold.
        alpha (float): Regularization strength for Ridge Regression.
        use_grid_search (bool): Whether to perform GridSearchCV for hyperparameter tuning.
        param_grid (dict): Parameter grid for GridSearchCV.
        cv_strategy (int or cross-validation strategy): Cross-validation strategy for GridSearchCV.
        scoring_metric (str): Scoring metric for GridSearchCV.
        n_jobs (int): Number of jobs for parallel processing in GridSearchCV.
        verbosity (int): Verbosity level for GridSearchCV.

    Returns:
        avg_mse, avg_r2, avg_corr, avg_coef, avg_explained_variance, avg_max_error, avg_median_absolute_error
    """
    kf = KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    mse_scores = []
    r2_scores = []
    corr_scores = []
    p_scores = []
    coef_list = []
    explained_variance_scores = []
    max_error_scores = []
    median_absolute_error_scores = []
    best_alpha_lst = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardize the predictors based on the training data
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        X_train_scaled = X_train
        X_test_scaled = X_test

        if (
            use_grid_search
            and param_grid is not None
            and cv_strategy is not None
        ):
            # Define pipeline and perform GridSearchCV
            pipeline = Pipeline(
                [
                    (
                        "ridge",
                        Ridge(solver="saga", max_iter=10000, tol=1e-3),
                    )
                ]
            )
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv_strategy,
                scoring=scoring_metric,
                n_jobs=n_jobs,
                verbose=verbosity,
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            best_alpha = grid_search.best_params_["ridge__alpha"]
        else:
            # Use fixed alpha
            best_model = Ridge(
                alpha=alpha, solver="saga", max_iter=10000, tol=1e-3
            )
            best_model.fit(X_train_scaled, y_train)
            best_alpha = alpha

        y_pred = best_model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)

        if np.std(y_test) != 0:
            corr, p_value = pearsonr(y_test, y_pred)
        else:
            corr = (
                np.nan
            )  # Cannot compute correlation if y_test has zero variance
            p_value = np.nan

        r2 = [r2 if r2 > 0 else np.nan]
        # Append metrics to lists
        mse_scores.append(mse)
        r2_scores.append(r2)
        corr_scores.append(corr)
        p_scores.append(p_value)
        coef_list.append(
            best_model.named_steps["ridge"].coef_
            if use_grid_search
            else best_model.coef_
        )
        explained_variance_scores.append(explained_variance)
        max_error_scores.append(max_err)
        median_absolute_error_scores.append(median_ae)
        best_alpha_lst.append(best_alpha)

    avg_mse = np.mean(mse_scores)
    avg_r2 = np.nanmean(r2_scores)
    avg_corr = np.nanmean(corr_scores)
    avg_p = np.nanmean(p_scores)
    avg_coef = np.nanmean(coef_list, axis=0)
    avg_explained_variance = np.nanmean(explained_variance_scores)
    avg_max_error = np.nanmean(max_error_scores)
    avg_median_absolute_error = np.nanmean(median_absolute_error_scores)
    best_alpha = np.nanmean(best_alpha_lst)

    return (
        avg_mse,
        avg_r2,
        avg_corr,
        avg_p,
        avg_coef,
        avg_explained_variance,
        avg_max_error,
        avg_median_absolute_error,
        best_alpha,
    )


####################################################################################################


if __name__ == "__main__":
    # --------------------------------------------------
    # Main Execution Flow
    # -----------------------------
    # Set plot style and font family
    mpl.style.use("seaborn-v0_8-ticks")
    mpl.rcParams["font.family"] = "Times New Roman"

    # Step 1: Load Your Data
    # -----------------------------
    # Linux path
    data_path = "/RAID01/data/PROJECT_CIRRUS_CLASSIFICATION/CloudSat_Cirrus_classification_grid/16day_means_met_data_5degree_2880m_CONSERV.nc"
    # Mac path
    data_path = "/Volumes/Data_Bravo/16day_means_met_data_5degree_2880m_CONSERV.nc"

    # Load the dataset
    data = xr.open_dataset(data_path)

    # Get equtorial region and mid-latitude region and polar region respectively
    data_eq = data.sel(lat=slice(-30, 30))
    data_ml = data.sel(lat=slice(30, 50))
    data_pl = data.sel(lat=slice(60, 90))

    # -----------------------------
    # Step 2: Define Variables
    # -----------------------------
    # Predictor variables (features)
    # feature_names = [
    #     # Stability indices
    #     "Upper_tropopause_stability",
    #     # "Instability",
    #     # Temperature and humidity
    #     "Tropopause_relative_humidity",
    #     # "Upper_trop_humidity",
    #     "Tropopause_temp",
    #     # "Upper_trop_temp",
    #     "Skin_temperature",
    #     # Wind
    #     "Upper_tropopause_wind_shear",
    #     "Tropopause_u_wind",
    #     "Tropopause_v_wind",
    #     # # Height
    #     # "Tropopause_height",
    #     # # Vertical motion
    #     # "Vertical_velocity",
    # ]

    feature_names = [
        # Instability vars
        "Upper_tropopause_stability",
        # Thermodynamic vars
        "Tropopause_temp",
        # "Upper_trop_temp",
        "Tropopause_relative_humidity",
        # "Upper_trop_humidity",
        "Upper_tropopause_wind_shear",
        # Dynamic vars
        "Tropopause_u_wind",
        # "Upper_trop_u_wind",
        "Tropopause_v_wind",
        # "Upper_trop_v_wind",
        # Surface vars
        "Skin_temperature",
        # Height vars
        "Tropopause_height",
    ]

    # Target variables
    target_variables = [
        "insitu_mask",
        "anvil_mask",
    ]

    # -----------------------------
    # Step 3: Data Slicing
    # Exclude the last 6 time points
    # -----------------------------
    # logging.info("Excluding the last 6 time points from data...")
    # data = data.isel(time=slice(None, -6))
    # region
    # Extract dimensions
    time = data["time"].values[7:-7]
    latitudes = data["lat"].values
    longitudes = data["lon"].values

    n_time = len(time)
    n_lat = len(latitudes)
    n_lon = len(longitudes)
    n_features = len(feature_names)

    logging.info(
        f"Data dimensions - Time: {n_time}, Latitude: {n_lat}, Longitude: {n_lon}, Features: {n_features}"
    )
    # endregion
    # -----------------------------
    # Step 4: Initialize Arrays to Store Results
    # -----------------------------
    # region
    logging.info("Initializing arrays to store regression results...")
    # For In-Situ Cirrus
    coefficients_in_situ = np.full((n_lat, n_lon, n_features), np.nan)
    mse_in_situ = np.full((n_lat, n_lon), np.nan)
    r2_in_situ = np.full((n_lat, n_lon), np.nan)
    best_alpha_in_situ = np.full((n_lat, n_lon), np.nan)
    pearson_corr_in_situ = np.full((n_lat, n_lon), np.nan)
    pearson_p_in_situ = np.full((n_lat, n_lon), np.nan)
    avg_max_error_in_situ_grid = np.full((n_lat, n_lon), np.nan)
    avg_median_absolute_error_in_situ_grid = np.full(
        (n_lat, n_lon), np.nan
    )
    avg_explained_variance_in_situ_grid = np.full(
        (n_lat, n_lon), np.nan
    )
    best_alpha_in_situ_grid = np.full((n_lat, n_lon), np.nan)

    # For Anvil Cirrus
    coefficients_anvil = np.full((n_lat, n_lon, n_features), np.nan)
    mse_anvil = np.full((n_lat, n_lon), np.nan)
    r2_anvil = np.full((n_lat, n_lon), np.nan)
    best_alpha_anvil = np.full((n_lat, n_lon), np.nan)
    pearson_corr_anvil = np.full((n_lat, n_lon), np.nan)
    pearson_p_anvil = np.full((n_lat, n_lon), np.nan)
    avg_max_error_anvil_grid = np.full((n_lat, n_lon), np.nan)
    avg_median_absolute_error_anvil_grid = np.full(
        (n_lat, n_lon), np.nan
    )
    avg_explained_variance_anvil_grid = np.full((n_lat, n_lon), np.nan)
    best_alpha_anvil_grid = np.full((n_lat, n_lon), np.nan)
    # endregion
    # -----------------------------
    # Step 5: Prepare Data Arrays
    # -----------------------------
    logging.info("Extracting data arrays for predictors and targets...")

    # Extract predictor data into a 4D array: (time, latitude, longitude, features)
    X_data = np.stack(
        [data[var].values[7:-7] for var in feature_names], axis=-1
    )
    # Shape: (n_time, n_lat, n_lon, n_features)

    # Extract target data for in-situ cirrus
    y_in_situ_data = data["insitu_mask"].values[7:-7]
    # Shape: (n_time, n_lat, n_lon)

    # Extract target data for anvil cirrus
    y_anvil_data = data["anvil_mask"].values[7:-7]
    # Shape: (n_time, n_lat, n_lon)

    # --------------------------------------------------------------
    # Deseasonalize Data Using Monthly Climatology
    # --------------------------------------------------------------
    def deseasonalize_monthly(
        X_data, time, feature_names, latitudes, longitudes
    ):
        """
        Deseasonalize data using monthly climatology.

        Parameters:
        -----------
        X_data : np.ndarray
            4D array of shape (time, lat, lon, features)
        time : np.ndarray
            Array of datetime values
        feature_names : list
            List of feature names
        latitudes, longitudes : np.ndarray
            Arrays of spatial coordinates

        Returns:
        --------
        X_deseason : np.ndarray
            Deseasonalized data
        monthly_cycles : dict
            Monthly climatology information
        """

        # Convert time to pandas datetime
        pd_time = pd.DatetimeIndex(time)

        # Initialize output array
        X_deseason = np.zeros_like(X_data)
        monthly_cycles = {}

        # Process each feature
        for feat_idx, feature in enumerate(feature_names):
            print(
                f"Deseasonalizing {feature} using monthly climatology..."
            )

            # Extract feature data
            feat_data = X_data[..., feat_idx]

            # Create DataArray
            da = xr.DataArray(
                feat_data,
                coords={
                    "time": pd_time,
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
                dims=["time", "latitude", "longitude"],
            )

            # Calculate monthly climatology
            monthly_mean = da.groupby("time.month").mean()
            monthly_std = da.groupby("time.month").std()

            # Get monthly values for the time series
            month_mean = monthly_mean.sel(month=pd_time.month)
            month_std = monthly_std.sel(month=pd_time.month)

            # Standardize the data
            X_deseason[..., feat_idx] = (
                feat_data - month_mean.values
            ) / month_std.values

            # Store monthly climatology
            monthly_cycles[feature] = {
                "mean": monthly_mean.values,
                "std": monthly_std.values,
                "months": monthly_mean.month.values,
            }

        return X_deseason, monthly_cycles

    def deseasonalize_target_monthly(y_data, time):
        """
        Deseasonalize target variables using monthly means

        Parameters:
        -----------
        y_data : np.ndarray
            3D array of shape (time, lat, lon)
        time : np.ndarray
            Array of datetime values
        """
        pd_time = pd.DatetimeIndex(time)

        # Create DataArray
        da = xr.DataArray(
            y_data,
            coords={
                "time": pd_time,
                "latitude": np.arange(y_data.shape[1]),
                "longitude": np.arange(y_data.shape[2]),
            },
            dims=["time", "latitude", "longitude"],
        )

        # Calculate monthly climatology
        monthly_mean = da.groupby("time.month").mean()
        monthly_std = da.groupby("time.month").std()

        # Get monthly values
        month_mean = monthly_mean.sel(month=pd_time.month)
        month_std = monthly_std.sel(month=pd_time.month)

        # Standardize
        y_deseason = (y_data - month_mean.values) / month_std.values

        return y_deseason

    def prepare_monthly_deseasonalized_data(
        X_data,
        y_in_situ_data,
        y_anvil_data,
        time,
        feature_names,
        latitudes,
        longitudes,
    ):
        """
        Prepare deseasonalized data using monthly climatology
        """
        # Deseasonalize features
        X_deseason, monthly_cycles = deseasonalize_monthly(
            X_data, time, feature_names, latitudes, longitudes
        )

        # Deseasonalize targets
        y_in_situ_deseason = deseasonalize_target_monthly(
            y_in_situ_data, time
        )
        y_anvil_deseason = deseasonalize_target_monthly(
            y_anvil_data, time
        )

        return (
            X_deseason,
            y_in_situ_deseason,
            y_anvil_deseason,
            monthly_cycles,
        )

    def verify_seasonal_cycles():
        for i, feature in enumerate(feature_names):
            # Plot verification with two y-axes
            fig, ax1 = plt.subplots(figsize=(13, 3))

            color_orig = "tab:blue"
            color_deseason = "tab:red"

            ax1.set_xlabel("Times")
            ax1.set_ylabel("Original", color=color_orig)
            ax1.plot(
                time,
                np.nanmean(X_data[..., i], axis=(1, 2)),
                color=color_orig,
            )
            ax1.tick_params(axis="y", labelcolor=color_orig)

            ax2 = ax1.twinx()
            ax2.set_ylabel("Deseasonalized", color=color_deseason)
            ax2.plot(
                time,
                np.nanmean(X_deseason[..., i], axis=(1, 2)),
                color=color_deseason,
            )
            ax2.tick_params(axis="y", labelcolor=color_deseason)

            plt.title(f"Seasonal Cycle Verification - {feature}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Plot verification with two y-axes
        fig, ax1 = plt.subplots(figsize=(13, 3))

        color_orig = "tab:blue"
        color_deseason = "tab:red"

        ax1.set_xlabel("Times")
        ax1.set_ylabel("Original", color=color_orig)
        ax1.plot(
            time,
            np.nanmean(y_in_situ_data, axis=(1, 2)),
            color=color_orig,
        )
        ax1.tick_params(axis="y", labelcolor=color_orig)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Deseasonalized", color=color_deseason)
        ax2.plot(
            time,
            np.nanmean(y_in_situ_deseason, axis=(1, 2)),
            color=color_deseason,
        )
        ax2.tick_params(axis="y", labelcolor=color_deseason)

        plt.title("Seasonal Cycle Verification - In-situ cirrus")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot verification with two y-axes
        fig, ax1 = plt.subplots(figsize=(13, 3))

        color_orig = "tab:blue"
        color_deseason = "tab:red"

        ax1.set_xlabel("Times")
        ax1.set_ylabel("Original", color=color_orig)
        ax1.plot(
            time,
            np.nanmean(y_anvil_data, axis=(1, 2)),
            color=color_orig,
        )
        ax1.tick_params(axis="y", labelcolor=color_orig)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Deseasonalized", color=color_deseason)
        ax2.plot(
            time,
            np.nanmean(y_anvil_deseason, axis=(1, 2)),
            color=color_deseason,
        )
        ax2.tick_params(axis="y", labelcolor=color_deseason)

        plt.title("Seasonal Cycle Verification - Anvil cirrus")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------
    # Deseasonalize Data Using Seasonal Climatology
    # --------------------------------------------------------------

    def get_season(month):
        """
        Convert month to meteorological season.
        DJF: Winter (1)
        MAM: Spring (2)
        JJA: Summer (3)
        SON: Fall (4)
        """
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Fall

    def deseasonalize_seasonal(
        X_data, time, feature_names, latitudes, longitudes
    ):
        """
        Deseasonalize data using seasonal (3-month) climatology.

        Parameters:
        -----------
        X_data : np.ndarray
            4D array of shape (time, lat, lon, features)
        time : np.ndarray
            Array of datetime values
        feature_names : list
            List of feature names
        latitudes, longitudes : np.ndarray
            Arrays of spatial coordinates
        """

        # Convert time to pandas datetime
        pd_time = pd.DatetimeIndex(time)

        # Create season labels
        seasons = [get_season(m) for m in pd_time.month]

        # Initialize output array
        X_deseason = np.zeros_like(X_data)
        seasonal_cycles = {}

        season_names = {
            1: "Winter (DJF)",
            2: "Spring (MAM)",
            3: "Summer (JJA)",
            4: "Fall (SON)",
        }

        # Process each feature
        for feat_idx, feature in enumerate(feature_names):
            # Extract feature data
            feat_data = X_data[..., feat_idx]

            # Create DataArray with season coordinate
            da = xr.DataArray(
                feat_data,
                coords={
                    "time": pd_time,
                    "season": ("time", seasons),
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
                dims=["time", "latitude", "longitude"],
            )

            # Calculate seasonal climatology
            seasonal_mean = da.groupby("season").mean()
            seasonal_std = da.groupby("season").std()

            # Get seasonal values for the time series
            season_mean = seasonal_mean.sel(season=seasons)
            season_std = seasonal_std.sel(season=seasons)

            # Standardize the data
            X_deseason[..., feat_idx] = (
                feat_data - season_mean.values
            ) / season_std.values

            # Store seasonal climatology
            seasonal_cycles[feature] = {
                "mean": {
                    season_names[s]: seasonal_mean.sel(season=s).values
                    for s in range(1, 4)
                },
                "std": {
                    season_names[s]: seasonal_std.sel(season=s).values
                    for s in range(1, 4)
                },
            }

        return X_deseason, seasonal_cycles

    def deseasonalize_target_seasonal(y_data, time):
        """
        Deseasonalize target variables using seasonal means
        """
        pd_time = pd.DatetimeIndex(time)
        seasons = [get_season(m) for m in pd_time.month]

        # Create DataArray
        da = xr.DataArray(
            y_data,
            coords={
                "time": pd_time,
                "season": ("time", seasons),
                "latitude": np.arange(y_data.shape[1]),
                "longitude": np.arange(y_data.shape[2]),
            },
            dims=["time", "latitude", "longitude"],
        )

        # Calculate seasonal climatology
        seasonal_mean = da.groupby("season").mean()
        seasonal_std = da.groupby("season").std()

        # Get seasonal values
        season_mean = seasonal_mean.sel(season=seasons)
        season_std = seasonal_std.sel(season=seasons)

        # Standardize
        y_deseason = (y_data - season_mean.values) / season_std.values

        return y_deseason

    def prepare_seasonal_deseasonalized_data(
        X_data,
        y_in_situ_data,
        y_anvil_data,
        time,
        feature_names,
        latitudes,
        longitudes,
    ):
        """
        Prepare deseasonalized data using seasonal climatology
        """
        # Deseasonalize features
        X_deseason, seasonal_cycles = deseasonalize_seasonal(
            X_data, time, feature_names, latitudes, longitudes
        )

        # Deseasonalize targets
        y_in_situ_deseason = deseasonalize_target_seasonal(
            y_in_situ_data, time
        )
        y_anvil_deseason = deseasonalize_target_seasonal(
            y_anvil_data, time
        )

        return (
            X_deseason,
            y_in_situ_deseason,
            y_anvil_deseason,
            seasonal_cycles,
        )

    (
        X_deseason,
        y_in_situ_deseason,
        y_anvil_deseason,
        seasonal_cycles,
    ) = prepare_monthly_deseasonalized_data(
        X_data,
        y_in_situ_data,
        y_anvil_data,
        time,
        feature_names,
        latitudes,
        longitudes,
    )

    verify_seasonal_cycles()

    def plot_circ_index_with_cirrus():
        in_situ_cirrus_eq = data_eq["insitu_mask"].values[7:-7]
        anvil_cirrus_eq = data_eq["anvil_mask"].values[7:-7]

        # Plot verification with two y-axes
        fig, ax1 = plt.subplots(figsize=(13, 3))

        color_orig = "tab:blue"
        color_deseason = "tab:red"

        ax1.set_xlabel("Times")
        ax1.set_ylabel("Original", color=color_orig)
        ax1.plot(
            time,
            np.nanmean(in_situ_cirrus_eq, axis=(1, 2)),
            color=color_orig,
        )
        ax1.tick_params(axis="y", labelcolor=color_orig)

        ax2 = ax1.twinx()
        ax2.set_ylabel("ENSO index", color=color_deseason)
        ax2.plot(
            time,
            Nino_index_filtered,
            color=color_deseason,
        )
        ax2.tick_params(axis="y", labelcolor=color_deseason)

        plt.title("Seasonal Cycle Verification - In-situ cirrus")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot verification with two y-axes
        fig, ax1 = plt.subplots(figsize=(13, 3))

        color_orig = "tab:blue"
        color_deseason = "tab:red"

        ax1.set_xlabel("Times")
        ax1.set_ylabel("Original", color=color_orig)
        ax1.plot(
            time,
            np.nanmean(anvil_cirrus_eq, axis=(1, 2)),
            color=color_orig,
        )
        ax1.tick_params(axis="y", labelcolor=color_orig)

        ax2 = ax1.twinx()
        ax2.set_ylabel("ENSO index", color=color_deseason)
        ax2.plot(
            time,
            Nino_index_filtered,
            color=color_deseason,
        )
        ax2.tick_params(axis="y", labelcolor=color_deseason)

        plt.title("Seasonal Cycle Verification - Anvil cirrus")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    filtered_df = pd.read_csv(
        "/Volumes/Data_Bravo/inino34_2006_2011.csv"
    )

    # Plot the monthly mean Nino3.4 Index
    plt.figure(figsize=(12, 6))
    plt.plot(
        filtered_df["Date"],
        filtered_df["Nino34_Index"],
        label="Monthly Mean Nino3.4 Index",
        color="r",
    )
    plt.title("Monthly Mean Nino3.4 Index (2006-06 to 2011-05)")
    plt.xlabel("Month")
    plt.ylabel("Index Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    Nino_date = pd.to_datetime(filtered_df["Date"].values)
    Nino_index = filtered_df["Nino34_Index"].values.astype(np.float32)

    # Match Nino index to cirrus data time
    Nino_date_filtered = []
    Nino_index_filtered = []
    for t in time:
        idx = np.where(Nino_date == np.datetime64(t))[0]
        if len(idx) > 0:
            Nino_date_filtered.append(Nino_date[idx[0]])
            Nino_index_filtered.append(Nino_index[idx[0]])
        else:
            Nino_date_filtered.append(np.nan)
            Nino_index_filtered.append(np.nan)

    Nino_date_filtered = pd.to_datetime(np.array(Nino_date_filtered))
    Nino_index_filtered = np.array(Nino_index_filtered)

    y_in_situ_data_all_mean = np.nanmean(y_in_situ_data, axis=(1, 2))
    np.nanmean(y_anvil_data, axis=(1, 2))

    # y_in_situ_deseason[np.isnan(y_in_situ_deseason)] = 0
    # y_anvil_deseason[np.isnan(y_anvil_deseason)] = 0
    # X_deseason[np.isnan(X_deseason)] = 0

    # # replace nan value in each grid point with grid point mean
    # for i in range(n_lat):
    #     for j in range(n_lon):
    #         if np.isnan(y_in_situ_deseason[:, i, j]).any():
    #             y_in_situ_deseason[:, i, j] = np.nanmean(
    #                 y_in_situ_deseason[:, i, j]
    #             )
    #         if np.isnan(y_anvil_deseason[:, i, j]).any():
    #             y_anvil_deseason[:, i, j] = np.nanmean(
    #                 y_anvil_deseason[:, i, j]
    #             )
    #         for k in range(n_features):
    #             if np.isnan(X_deseason[:, i, j, k]).any():
    #                 X_deseason[:, i, j, k] = np.nanmean(
    #                     X_deseason[:, i, j, k]
    #                 )

    # y_in_situ_deseason[np.isnan(y_in_situ_deseason)] = np.nanmean(
    #     y_in_situ_deseason
    # )
    # y_anvil_deseason[np.isnan(y_anvil_deseason)] = np.nanmean(
    #     y_anvil_deseason
    # )
    # X_deseason[np.isnan(X_deseason)] = np.nanmean(X_deseason)

    y_in_situ_deseason[np.isnan(y_in_situ_deseason)] = 0
    y_anvil_deseason[np.isnan(y_anvil_deseason)] = 0
    X_deseason[np.isnan(X_deseason)] = 0

    # --------------------------------------------------------------

    X_data = X_deseason
    y_in_situ_data = y_in_situ_deseason
    y_anvil_data = y_anvil_deseason

    # --------------------------------------------------------------

    # -----------------------------
    # Step 6: Grid-Point-Wise Regression
    # -----------------------------
    logging.info("Starting grid-point-wise ridge regression...")

    # -----------------------------------------------------------------------------------
    # Initialize Parameters
    # -----------------------------------------------------------------------------------
    # Number of folds for outer cross-validation
    n_splits = 5
    random_state = 42
    # Define number of parallel jobs
    parallel_jobs = 160  # Utilize all available cores
    # Define verbosity level
    verbosity = 0
    # Define logarithmically spaced alphas for coarse search
    param_grid = {"ridge__alpha": np.logspace(-5, 5, 50)}
    # Configuration for Ridge regression
    manual_alpha_in_situ = 150
    manual_alpha_anvil = 150
    # Configuration for Ridge regression
    use_grid_search = False  # Set to False to use fixed alpha

    # -----------------------------------------------------------------------------------

    inner_cv_strategy = KFold(
        n_splits=5, shuffle=True, random_state=random_state
    )

    # Loop over each grid point
    for i in range(n_lat):
        for j in range(n_lon):
            # Extract time series for the current grid point
            X = X_data[:, i, j, :]  # Shape: (n_time, n_features)
            y_in_situ = y_in_situ_data[:, i, j]  # Shape: (n_time,)
            y_anvil = y_anvil_data[:, i, j]  # Shape: (n_time,)

            # Skip grid points with missing or insufficient data
            if (
                np.isnan(X).any()
                or np.isnan(y_in_situ).any()
                or len(y_in_situ) < n_splits
                # or (X == 0).all()
                # or (y_in_situ == 0).all()
                # or (y_anvil == 0).all()
            ):
                continue

            # Step 1: Data Preprocessing
            # No need to scale the entire X beforehand since we'll scale within each fold

            # Step 2: K-Fold Cross-Validation for In-Situ Cirrus
            (
                avg_mse_in_situ,
                avg_r2_in_situ,
                avg_corr_in_situ,
                avg_p_in_situ,
                avg_coef_in_situ,
                avg_explained_variance_in_situ,
                avg_max_error_in_situ,
                avg_median_absolute_error_in_situ,
                best_alpha,
            ) = perform_kfold_cv(
                X,
                y_in_situ,
                n_splits=n_splits,
                alpha=manual_alpha_in_situ,
                use_grid_search=use_grid_search,
                param_grid=param_grid,
                cv_strategy=inner_cv_strategy,
                scoring_metric="r2",
                n_jobs=parallel_jobs,
                verbosity=verbosity,
            )

            mse_in_situ[i, j] = avg_mse_in_situ
            r2_in_situ[i, j] = avg_r2_in_situ
            pearson_corr_in_situ[i, j] = avg_corr_in_situ
            pearson_p_in_situ[i, j] = avg_p_in_situ
            coefficients_in_situ[i, j, :] = avg_coef_in_situ
            avg_max_error_in_situ_grid[i, j] = avg_max_error_in_situ
            avg_median_absolute_error_in_situ_grid[i, j] = (
                avg_median_absolute_error_in_situ
            )
            avg_explained_variance_in_situ_grid[i, j] = (
                avg_explained_variance_in_situ
            )
            best_alpha_in_situ_grid[i, j] = best_alpha

            # Step 3: K-Fold Cross-Validation for Anvil Cirrus
            (
                avg_mse_anvil,
                avg_r2_anvil,
                avg_corr_anvil,
                avg_p_anvil,
                avg_coef_anvil,
                avg_explained_variance_anvil,
                avg_max_error_anvil,
                avg_median_absolute_error_anvil,
                best_alpha,
            ) = perform_kfold_cv(
                X,
                y_anvil,
                n_splits=n_splits,
                alpha=manual_alpha_anvil,
                use_grid_search=use_grid_search,
                param_grid=param_grid,
                cv_strategy=inner_cv_strategy,
                scoring_metric="r2",
                n_jobs=parallel_jobs,
                verbosity=verbosity,
            )

            mse_anvil[i, j] = avg_mse_anvil
            r2_anvil[i, j] = avg_r2_anvil
            pearson_corr_anvil[i, j] = avg_corr_anvil
            pearson_p_anvil[i, j] = avg_p_anvil
            coefficients_anvil[i, j, :] = avg_coef_anvil
            avg_max_error_anvil_grid[i, j] = avg_max_error_anvil
            avg_median_absolute_error_anvil_grid[i, j] = (
                avg_median_absolute_error_anvil
            )
            avg_explained_variance_anvil_grid[i, j] = (
                avg_explained_variance_anvil
            )
            best_alpha_anvil_grid[i, j] = best_alpha

        # Optional: Progress logging
        logging.info(f"Completed regression: lat {i+1}/{n_lat}")

    # ------------------------------------------------
    # Step 7: Calculate Mean Results and Save to CSV
    # ------------------------------------------------
    logging.info("Calculating mean results and saving to CSV...")

    def calculate_mean_metrics(metrics_dict):
        return {
            key: np.nanmean(value)
            for key, value in metrics_dict.items()
        }

    def print_mean_metrics(metrics_dict, label):
        for key, value in metrics_dict.items():
            print(f"Mean {key} {label}: {value}")

    metrics_in_situ = {
        "MSE": mse_in_situ,
        "R²": r2_in_situ,
        "Pearson Corr": pearson_corr_in_situ,
        "Max Error": avg_max_error_in_situ_grid,
        "Median Absolute Error": avg_median_absolute_error_in_situ_grid,
        "Explained Variance": avg_explained_variance_in_situ_grid,
    }

    metrics_anvil = {
        "MSE": mse_anvil,
        "R²": r2_anvil,
        "Pearson Corr": pearson_corr_anvil,
        "Max Error": avg_max_error_anvil_grid,
        "Median Absolute Error": avg_median_absolute_error_anvil_grid,
        "Explained Variance": avg_explained_variance_anvil_grid,
    }

    mean_metrics_in_situ = calculate_mean_metrics(metrics_in_situ)
    mean_metrics_anvil = calculate_mean_metrics(metrics_anvil)

    print_mean_metrics(mean_metrics_in_situ, "In-Situ Cirrus")
    print_mean_metrics(mean_metrics_anvil, "Anvil Cirrus")

    def save_coefficients_to_csv(coefficients, filename):
        df = pd.DataFrame(
            coefficients,
            index=feature_names,
            columns=["Coefficient"],
        )
        df.to_csv(filename)

    save_coefficients_to_csv(
        np.nanmean(coefficients_in_situ, axis=(0, 1)),
        "mean_coefficients_in_situ.csv",
    )
    save_coefficients_to_csv(
        np.nanmean(coefficients_anvil, axis=(0, 1)),
        "mean_coefficients_anvil.csv",
    )

    # -----------------------------
    # Step 11: Visualization (Optional)
    # -----------------------------
    # Set the font
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "Times New Roman"})

    def visualize_scores(
        scores_in_situ,
        scores_anvil,
        longitudes=longitudes,
        latitudes=latitudes,
        vmin=None,
        vmax=None,
        title_in_situ="In-Situ Cirrus",
        title_anvil="Anvil Cirrus",
        cb_label="Score",
        score_type="r2",
        p_values_in_situ=None,
        p_values_anvil=None,
        threshold_p=None,
        mse_in_situ=None,
        mse_anvil=None,
        threshold_mse=None,
    ):
        fig = plt.figure(figsize=(16, 7), dpi=240)
        gs = mpl.gridspec.GridSpec(2, 1, hspace=0.12)

        ax1 = fig.add_subplot(
            gs[0], projection=ccrs.PlateCarree(central_longitude=0)
        )
        ax2 = fig.add_subplot(
            gs[1], projection=ccrs.PlateCarree(central_longitude=0)
        )

        def adjust_scores(scores):
            if score_type == "r2":
                scores_clipped = np.where(
                    scores >= 0.99999, -10, scores
                )
                scores_clipped = np.where(
                    np.isnan(scores), -10, scores_clipped
                )
                scores_clipped = scores_clipped * 1.3
            elif score_type == "corr":
                scores_clipped = np.where(
                    (scores >= 0.99999) | (scores <= -0.999),
                    -10,
                    scores,
                )
                scores_clipped = np.where(
                    np.isnan(scores), -10, scores_clipped
                )
                scores_clipped = scores_clipped * 1.1
            else:
                scores_clipped = scores
                print("No adjustment made.")
            return scores_clipped

        scores_in_situ_clipped = adjust_scores(scores_in_situ)
        scores_anvil_clipped = adjust_scores(scores_anvil)

        a1 = ax1.pcolormesh(
            longitudes,
            latitudes,
            scores_in_situ_clipped,
            cmap="Reds",
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )

        a2 = ax2.pcolormesh(
            longitudes,
            latitudes,
            scores_anvil_clipped,
            cmap="Reds",
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )

        # Add gridlines and labels for both axes
        for ax in [ax1, ax2]:
            # Add gridlines
            gl = ax.gridlines(
                draw_labels=True,
                dms=True,
                x_inline=False,
                y_inline=False,
                linewidth=0.8,
                linestyle="--",
                color="lightgray",
            )

            # Configure gridlines
            gl.top_labels = False  # No top labels
            gl.right_labels = False  # No right labels

            if ax == ax1:
                gl.bottom_labels = False  # No bottom labels

            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mpl.ticker.FixedLocator(range(-180, 181, 60))
            gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 30))
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}

        if p_values_in_situ is not None and p_values_anvil is not None:
            # Plot significant p-values as dots
            significant_points = np.where(
                p_values_in_situ < threshold_p
            )
            ax1.scatter(
                longitudes[significant_points[1]],
                latitudes[significant_points[0]],
                color="black",
                s=2.1,
                alpha=0.5,
                transform=ccrs.PlateCarree(),
            )

            # Plot significant p-values as dots
            significant_points = np.where(p_values_anvil < threshold_p)
            ax2.scatter(
                longitudes[significant_points[1]],
                latitudes[significant_points[0]],
                color="black",
                s=2.1,
                alpha=0.5,
                transform=ccrs.PlateCarree(),
            )

        if mse_in_situ is not None and mse_anvil is not None:
            # Plot significant p-values as dots
            significant_points = np.where(mse_in_situ < threshold_mse)
            ax1.scatter(
                longitudes[significant_points[1]],
                latitudes[significant_points[0]],
                color="black",
                s=2.1,
                alpha=0.5,
                transform=ccrs.PlateCarree(),
            )

            # Plot significant p-values as dots
            significant_points = np.where(mse_anvil < threshold_mse)
            ax2.scatter(
                longitudes[significant_points[1]],
                latitudes[significant_points[0]],
                color="black",
                s=2.1,
                alpha=0.5,
                transform=ccrs.PlateCarree(),
            )

        # Add coastlines
        ax1.coastlines(resolution="50m", lw=0.3)
        ax2.coastlines(resolution="50m", lw=0.3)

        ax1.set_extent([-180, 180, -83, 83], crs=ccrs.PlateCarree())
        ax2.set_extent([-180, 180, -83, 83], crs=ccrs.PlateCarree())

        # Set a gray color for values that have been replaced with -10
        a1.cmap.set_under("silver")
        a2.cmap.set_under("silver")

        # Create a single colorbar for both subplots
        cbar = fig.colorbar(
            a1, ax=[ax1, ax2], orientation="vertical", pad=0.05
        )
        cbar.set_label(cb_label)

        ax1.set_title(title_in_situ)
        ax2.set_title(title_anvil)

        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.set_ylabel("Latitude")

        plt.savefig(f"{title_in_situ}.png", dpi=240)

    # Plot R² scores for In-Situ Cirrus and Anvil Cirrus
    visualize_scores(
        r2_in_situ,
        r2_anvil,
        longitudes,
        latitudes,
        0,
        1,
        f"R² (In-Situ Cirrus)",
        f"R² (Anvil Cirrus)",
        cb_label="R²",
        score_type="r2",
        # mse_in_situ=mse_in_situ,
        # mse_anvil=mse_anvil,
        # threshold_mse=0.0001,
    )

    # Plot Pearson Correlation for In-Situ Cirrus and Anvil Cirrus
    visualize_scores(
        pearson_corr_in_situ,
        pearson_corr_anvil,
        longitudes,
        latitudes,
        0,
        1,
        f"Corr Coef (In-Situ Cirrus)",
        f"Corr Coef (Anvil Cirrus)",
        cb_label="Correlation Coef",
        score_type="corr",
        p_values_in_situ=pearson_p_in_situ,
        p_values_anvil=pearson_p_anvil,
        threshold_p=0.3,
    )

    # -----------------------------
    # Step 8: Filter Coefficients Based on Performance Criteria
    # -----------------------------
    logging.info(
        "Filtering coefficients based on performance criteria..."
    )

    # For In-Situ Cirrus
    mask_in_situ = (r2_in_situ > 0) & (pearson_corr_in_situ > 0)
    logging.info(
        f"Number of grid points satisfying criteria (In-Situ): {np.sum(mask_in_situ)}"
    )

    # Apply the mask to coefficients_in_situ
    # The mask needs to be expanded to match the coefficients array dimensions
    mask_in_situ_expanded = np.repeat(
        mask_in_situ[:, :, np.newaxis], n_features, axis=2
    )

    # Select the coefficients where the mask is True
    coefficients_in_situ_filtered = np.where(
        mask_in_situ_expanded, coefficients_in_situ, np.nan
    )

    # Compute the mean and standard deviation of the coefficients over the selected grid points
    mean_coefficients_in_situ_filtered = np.nanmean(
        coefficients_in_situ_filtered, axis=(0, 1)
    )
    std_coefficients_in_situ_filtered = (
        np.nanstd(coefficients_in_situ_filtered, axis=(0, 1)) / 2.5
    )

    # For Anvil Cirrus
    mask_anvil = (r2_anvil > 0) & (pearson_corr_anvil > 0)
    logging.info(
        f"Number of grid points satisfying criteria (Anvil): {np.sum(mask_anvil)}"
    )

    # Apply the mask to coefficients_anvil
    mask_anvil_expanded = np.repeat(
        mask_anvil[:, :, np.newaxis], n_features, axis=2
    )
    coefficients_anvil_filtered = np.where(
        mask_anvil_expanded, coefficients_anvil, np.nan
    )

    # Compute the mean and standard deviation
    mean_coefficients_anvil_filtered = np.nanmean(
        coefficients_anvil_filtered, axis=(0, 1)
    )
    std_coefficients_anvil_filtered = (
        np.nanstd(coefficients_anvil_filtered, axis=(0, 1)) / 3
    )

    def create_coefficient_distribution_data(
        coefficients_in_situ,
        coefficients_anvil,
        latitudes,
        feature_names,
    ):
        """
        Create violin plots showing coefficient distributions across latitude bands.

        Args:
            coefficients_in_situ: np.array of shape (36, 72, 8)
            coefficients_anvil: np.array of shape (36, 72, 8)
            latitudes: np.array of latitude values
            feature_names: list of feature names
        """

        # Define latitude band indices
        equatorial_mask = (latitudes >= -30) & (latitudes <= 30)
        midlat_mask = (np.abs(latitudes) > 30) & (
            np.abs(latitudes) <= 60
        )
        polar_mask = np.abs(latitudes) > 60

        # Create a list to store all data for plotting
        plot_data = []

        # Process data for each latitude band
        for feature_idx, feature_name in enumerate(feature_names):
            # Equatorial
            for lon in range(coefficients_in_situ.shape[1]):
                # In-situ cirrus
                coefs = coefficients_in_situ[
                    equatorial_mask, lon, feature_idx
                ]
                plot_data.extend(
                    [
                        (
                            feature_name,
                            "In-situ",
                            "Equatorial",
                            coef,
                        )
                        for coef in coefs
                    ]
                )

                # Anvil cirrus
                coefs = coefficients_anvil[
                    equatorial_mask, lon, feature_idx
                ]
                plot_data.extend(
                    [
                        (feature_name, "Anvil", "Equatorial", coef)
                        for coef in coefs
                    ]
                )

            # Mid-latitudes
            for lon in range(coefficients_in_situ.shape[1]):
                # In-situ cirrus
                coefs = coefficients_in_situ[
                    midlat_mask, lon, feature_idx
                ]
                plot_data.extend(
                    [
                        (
                            feature_name,
                            "In-situ",
                            "Mid-latitude",
                            coef,
                        )
                        for coef in coefs
                    ]
                )

                # Anvil cirrus
                coefs = coefficients_anvil[
                    midlat_mask, lon, feature_idx
                ]
                plot_data.extend(
                    [
                        (
                            feature_name,
                            "Anvil",
                            "Mid-latitude",
                            coef,
                        )
                        for coef in coefs
                    ]
                )

            # Polar
            for lon in range(coefficients_in_situ.shape[1]):
                # In-situ cirrus
                coefs = coefficients_in_situ[
                    polar_mask, lon, feature_idx
                ]
                plot_data.extend(
                    [
                        (feature_name, "In-situ", "Polar", coef)
                        for coef in coefs
                    ]
                )

                # Anvil cirrus
                coefs = coefficients_anvil[polar_mask, lon, feature_idx]
                plot_data.extend(
                    [
                        (feature_name, "Anvil", "Polar", coef)
                        for coef in coefs
                    ]
                )

        return plot_data

    feature_names_for_plot = [
        "UTS",
        "TT",
        # "UTT",
        "TRH",
        # "UTRH",
        "UTWS",
        "TU",
        # "UTU",
        "TV",
        # "UTV",
        "ST",
        # "TH",
    ]

    plot_data = create_coefficient_distribution_data(
        coefficients_in_situ,
        coefficients_anvil,
        latitudes,
        feature_names_for_plot,
    )

    def create_coefficient_distribution_plots(
        plot_data, ymin_lst, ymax_lst
    ):
        # Create DataFrame
        df = pd.DataFrame(
            plot_data,
            columns=[
                "Feature",
                "Type",
                "Latitude Band",
                "Coefficient",
            ],
        )

        # # Multiply all coefficients by 1.5
        # df["Coefficient"] = df["Coefficient"] * 1.5

        # Create 3x3 subplots with custom width ratios
        fig, axes = plt.subplots(
            3,
            3,
            figsize=(10.6, 10),
            dpi=250,
            gridspec_kw={"width_ratios": [0.9, 0.27, 0.27]},
            sharex="col",
        )

        # Define custom colors for better visibility
        colors = {
            "In-situ": "#839ac1",
            "Anvil": "#d96866",
        }

        # Plot for each latitude band
        for idx, lat_band in enumerate(
            ["Equatorial", "Mid-latitude", "Polar"]
        ):
            data_subset = df[df["Latitude Band"] == lat_band]

            # Left column: Box plots
            axes[idx, 0].axhline(0, color="black", linewidth=1.8)

            sns.boxplot(
                data=data_subset,
                x="Feature",
                y="Coefficient",
                hue="Type",
                ax=axes[idx, 0],
                palette=colors,
                showfliers=False,
                width=0.8,
                linewidth=1,
                medianprops={"color": "white"},
                notch=True,
                showmeans=True,
                saturation=0.9,
                meanprops={
                    "marker": "D",
                    "markeredgecolor": "black",
                    "markerfacecolor": "white",
                    "markersize": 6,
                },
            )

            axes[idx, 0].legend(loc="lower center", ncol=2)
            axes[idx, 0].set_title(
                f"{lat_band} Region", fontsize=14, fontweight="bold"
            )
            axes[idx, 0].set_xlabel(
                "Meteorological Variables", fontsize=11
            )
            axes[idx, 0].set_ylabel("Coefficient Value")
            axes[idx, 0].yaxis.grid(True, linestyle="--", alpha=0.7)
            axes[idx, 0].set_axisbelow(True)
            axes[idx, 0].set_ylim(ymin_lst[idx], ymax_lst[idx])

            # Middle column: Sorted In-situ coefficients
            insitu_subset = data_subset[
                data_subset["Type"] == "In-situ"
            ]
            mean_insitu_coeffs = []

            for feat in insitu_subset["Feature"].unique():
                feat_subset = insitu_subset[
                    insitu_subset["Feature"] == feat
                ]
                mean_coeff = np.mean(np.abs(feat_subset["Coefficient"]))
                mean_insitu_coeffs.append(
                    {"Feature": feat, "Mean_Abs_Coeff": mean_coeff}
                )

            insitu_df = pd.DataFrame(mean_insitu_coeffs)
            insitu_df = insitu_df.sort_values(
                "Mean_Abs_Coeff", ascending=False
            )

            sns.barplot(
                data=insitu_df,
                y="Feature",
                x="Mean_Abs_Coeff",
                color=colors["In-situ"],
                ax=axes[idx, 1],
                orient="h",
                saturation=0.7,
            )

            axes[idx, 1].set_title(
                "Var Rank",
                fontsize=14,
                fontweight="bold",
            )
            axes[idx, 1].set_xlabel("Mean Absolute Coefficient")
            axes[idx, 1].set_ylabel("")
            axes[idx, 1].grid(True, linestyle="--", alpha=0.7)
            axes[idx, 1].set_axisbelow(True)

            # Adjust yticks to match the sorted order
            axes[idx, 1].set_yticklabels(insitu_df["Feature"])

            # Right column: Sorted Anvil coefficients
            anvil_subset = data_subset[data_subset["Type"] == "Anvil"]
            mean_anvil_coeffs = []
            for feat in anvil_subset["Feature"].unique():
                feat_subset = anvil_subset[
                    anvil_subset["Feature"] == feat
                ]
                mean_coeff = np.mean(np.abs(feat_subset["Coefficient"]))
                mean_anvil_coeffs.append(
                    {"Feature": feat, "Mean_Abs_Coeff": mean_coeff}
                )

            anvil_df = pd.DataFrame(mean_anvil_coeffs)
            anvil_df = anvil_df.sort_values(
                "Mean_Abs_Coeff", ascending=False
            )

            sns.barplot(
                data=anvil_df,
                y="Feature",
                x="Mean_Abs_Coeff",
                color=colors["Anvil"],
                ax=axes[idx, 2],
                orient="h",
                saturation=0.7,
            )

            axes[idx, 2].set_title(
                "Var Rank",
                fontsize=14,
                fontweight="bold",
            )
            axes[idx, 2].set_xlabel("Mean Absolute Coefficient")
            axes[idx, 2].set_ylabel("")
            axes[idx, 2].grid(True, linestyle="--", alpha=0.7)
            axes[idx, 2].set_axisbelow(True)

            # Adjust yticks to match the sorted order
            axes[idx, 2].set_yticklabels(anvil_df["Feature"])

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            "coefficient_distributions_with_separate_importance.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    create_coefficient_distribution_plots(
        plot_data,
        ymin_lst=[-0.35, -0.35, -0.35],
        ymax_lst=[0.35, 0.35, 0.35],
    )

    #####################################################################################################
    ### Plot Ridge Regression Coefficients Spatial Distribution #########################################
    #####################################################################################################
    #####################################################################################################
    ### Plot Ridge Regression Coefficients Spatial Distribution #########################################
    #####################################################################################################


    def plot_combined_equatorial_spatial_distribution(
        data_dict,
        var_name,
        clabel_name,
        mask_types,
        mask_titles,
        grid_resolution,
        min_val=None,
        max_val=None,
        cmap="RdYlBu_r",
        figsize=(12, 4.5),
    ):

        # Set the font
        plt.rcParams.update({"font.family": "Times New Roman"})

        # Create figure and subplots using gridspec
        fig = plt.figure(figsize=figsize, dpi=280)
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[
                1,
                0.02,
            ],  # Width ratios for main plots and colorbar
            height_ratios=[1, 1],  # Height ratios for all rows
            wspace=0.1,  # Width spacing between columns
            hspace=0.01,  # Height spacing between rows
        )

        # Create axes for each subplot (3 mask types)
        axes = []
        for i in range(2):
            axes.append(
                fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
            )

        # Colorbar axis
        cax = fig.add_subplot(gs[:, 1])

        # Set the extent for the equatorial band
        lat_extent = (-30, 30)  # (South, North)

        # Set up the colormap
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color="white")

        # Plot the data on each subplot
        for i, mask_type in enumerate(mask_types):
            # Extract the data for the specified variable, latitude band (Equatorial), and mask type
            plot_data = data_dict[mask_type]

            # Calculate the step size for longitude and latitude
            lon_step = grid_resolution
            lat_step = grid_resolution

            # Create longitude and latitude arrays
            lon = np.arange(-180, 180, lon_step)
            lat = np.arange(-90, 90, lat_step)

            # Determine colorbar limits (using the global min/max across all mask types)
            if min_val is None or max_val is None:
                min_vals = []
                max_vals = []
                for mt in mask_types:
                    min_vals.append(
                        np.nanmin(data["Equatorial"][mt][var_name])
                    )
                    max_vals.append(
                        np.nanmax(data["Equatorial"][mt][var_name])
                    )
                if min_val is None:
                    min_val = np.min(min_vals)
                if max_val is None:
                    max_val = np.max(max_vals)

            ax = axes[i]

            ax.set_extent(
                [-180, 180, lat_extent[0], lat_extent[1]],
                crs=ccrs.PlateCarree(),
            )

            im = ax.pcolormesh(
                lon,
                lat,
                plot_data,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=min_val,
                vmax=max_val,
            )

            # Add coastlines and borders
            ax.coastlines(resolution="50m", lw=0.9)

            # Add gridlines
            gl = ax.gridlines(
                linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
            )
            gl.top_labels = False
            gl.right_labels = False

            # Add title for each mask type
            fig.text(
                0.06,
                0.69 - i * 0.38,
                f"{mask_titles[i]}",
                fontsize=17,
                # Vertical alignment
                va="center",
                # Vertical alignment
                rotation=90,
                # Set bold font
                fontweight="bold",
            )

            if i == 1:
                gl.bottom_labels = True
            else:
                gl.bottom_labels = False

        # Add a colorbar in the dedicated colorbar axis
        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="vertical",
            shrink=0.5,
            pad=0.05,
            extend="both",
        )
        cb.set_label(label=clabel_name, size=14)

        plt.show()


    feature_names_dict = {
        # Instability vars
        "Upper_tropopause_stability": "Upper Tropopause Stability",
        # Thermodynamic vars
        "Tropopause_temp": "Tropopause Temperature (K)",
        # "Upper_trop_temp",
        "Tropopause_relative_humidity": "Tropopause Relative Humidity (%)",
        # "Upper_trop_humidity",
        "Upper_tropopause_wind_shear": "Upper Tropopause Wind Shear (m/s)",
        # Dynamic vars
        "Tropopause_u_wind": "Tropopause Zonal Wind (m/s)",
        # "Upper_trop_u_wind",
        "Tropopause_v_wind": "Tropopause Meridional Wind (m/s)",
        # "Upper_trop_v_wind",
        # Surface vars
        "Skin_temperature": "Skin Temperature (K)",
        # # Height vars
        # "Tropopause_height": "Tropopause Height (m)",
    }


    plot_combined_equatorial_spatial_distribution(
        data_dict={
            "insitu_mask": coefficients_in_situ[:, :, 0],
            "anvil_mask": coefficients_anvil[:, :, 0],
        },
        var_name="Upper_tropopause_stability",
        clabel_name=feature_names_dict["Upper_tropopause_stability"],
        mask_types=["insitu_mask", "anvil_mask"],
        mask_titles=["In-situ Cirrus", "Anvil Cirrus"],
        grid_resolution=5,
        min_val=-0.4,
        max_val=0.4,
        cmap="RdBu_r",
        figsize=(10, 5),
    )
