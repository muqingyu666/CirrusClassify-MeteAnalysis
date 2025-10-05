# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-10-05 07:46:12

import gc
import cartopy.crs as ccrs
import warnings
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import sys

from muqy_20240710_util_cirrus_class_freq_micro import (
    load_cirrus_data,
    process_cloudsat_data,
    plot_dual_hemisphere_self_cmap,
    cld_3d_sructure_plot_fig0,
    cld_3d_sructure_plot_fig1,
)

import importlib

# Reload the module to reflect any changes made to it
importlib.reload(sys.modules["muqy_20240710_util_cirrus_class_freq_micro"])


# Set matplotlib style
mpl.style.use("ggplot")
mpl.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "Times New Roman"

# Disable all warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------
# Visualization Functions


def plot_dual_hemisphere_dual_cbar_self_cmap(
    data1,
    data2,
    min_max_pair1,
    min_max_pair2,
    cb_label,
    title1,
    title2,
    cmap="RdYlBu_r",
    grid_shape=(90, 180),
):
    # Set the font
    plt.rcParams.update({"font.family": "Times New Roman"})

    # Calculate the step size for longitude and latitude
    lon_step = 360 / grid_shape[1]
    lat_step = 180 / grid_shape[0]

    # Create longitude and latitude arrays
    lon = np.linspace(-180, 180 - lon_step, grid_shape[1], dtype=np.float32)
    lat = np.linspace(-90, 90 - lat_step, grid_shape[0], dtype=np.float32)

    # Create figure and gridspec
    fig = plt.figure(figsize=(18, 4), dpi=280)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.05, 1, 0.05])

    # Create subplots
    ax1 = fig.add_subplot(
        gs[0], projection=ccrs.PlateCarree(central_longitude=0)
    )
    ax2 = fig.add_subplot(
        gs[2], projection=ccrs.PlateCarree(central_longitude=0)
    )
    cax1 = fig.add_subplot(gs[1])
    cax2 = fig.add_subplot(gs[3])

    # vmin and vmax values
    min_val1, max_val1 = min_max_pair1
    min_val2, max_val2 = min_max_pair2

    # Set up the colormap
    cmap = plt.get_cmap(cmap)
    cmap.set_under(color="white")
    cmap.set_bad(color="grey")

    for ax, data, title, cax, min_val, max_val in zip(
        [ax1, ax2],
        [data1, data2],
        [title1, title2],
        [cax1, cax2],
        [min_val1, min_val2],
        [max_val1, max_val2],
    ):
        ax.set_extent([-180, 180, -81, 81], crs=ccrs.PlateCarree())

        # Plot the data
        im = ax.pcolormesh(
            lon,
            lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=min_val,
            vmax=max_val,
        )

        # Add coastlines
        ax.coastlines(resolution="50m", lw=0.5)

        # Add title
        ax.set_title(title)

        # Add gridlines
        gl = ax.gridlines(linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True)
        gl.top_labels = False

        # Only show y labels for the left subplot
        if ax == ax2:
            gl.left_labels = False

        # Add a colorbar that spans both subplots
        cb = fig.colorbar(im, cax=cax, shrink=0.3)
        cb.set_label(label=cb_label)

    plt.tight_layout()
    plt.show()


def plot_cld_vertical_profile_lat_band(
    data_pair1,
    data_pair2,
    data_pair3,
    xlabel1,
    xlabel2,
    xlabel3,
):
    # Set the font
    plt.rcParams.update({"font.family": "Times New Roman"})

    # Create figure and gridspec
    fig = plt.figure(figsize=(6, 6), dpi=250)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    # Extract the data pair, data1 is insitu and data2 is anvil
    data_insitu_1, data_anvil_1 = data_pair1
    data_insitu_2, data_anvil_2 = data_pair2
    data_insitu_3, data_anvil_3 = data_pair3

    # Create subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax, data, xlabel in zip(
        [ax1, ax2, ax3],
        [
            (data_insitu_1, data_anvil_1),
            (data_insitu_2, data_anvil_2),
            (data_insitu_3, data_anvil_3),
        ],
        [xlabel1, xlabel2, xlabel3],
    ):
        # Unpack the data
        data_insitu, data_anvil = data

        ax.plot(data_insitu, heights, label="In-situ Cirrus")
        ax.plot(data_anvil, heights, label="Anvil Cirrus")

        ax.set_xlabel(xlabel)
        ax.legend(loc="upper right")

        # Only show y labels for the left subplot
        if ax != ax1:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Height (km)")

    plt.tight_layout()
    plt.show()


##############################################################################################################

# Main execution
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize

    def cld_3d_sructure_plot_fig0(
        cloud_frequency_insitu,
        cloud_frequency_anvil,
        zticks=np.arange(0, 70, 20),
        zlim=slice(0, 50),
        elev=110,
        azim=-170,
        roll=-80,
        vmin=0,
        vmax=60,
        # Plot flag to control whether tropopause is plotted
        plot_flag_trop=True,
        # Plot flag to control which heights are plotted
        # 0 means 3km below tropopause, 1 means 300 hPa, 2 means 200 hPa below tropopause
        plot_flag=0,
        **kwargs,
    ):
        cloud_frequency_insitu_surf = np.copy(cloud_frequency_insitu)
        cloud_frequency_insitu_surf[cloud_frequency_insitu_surf == 0] = (
            -0.0000001
        )

        # Set up the plot
        fig, ax = plt.subplots(
            subplot_kw=dict(projection="3d"),
            figsize=(9.5, 9.5),
            dpi=330,
            tight_layout=True,
        )

        # Light source object for hillshading
        lat_mesh, height_mesh = np.meshgrid(
            kwargs["latitudes"], kwargs["heights"]
        )

        # Plot the mean profil of each cloud type
        ax.plot(
            kwargs["heights"],
            # insitu_mean_vertical_profile * 1.55,
            np.nanmax(cloud_frequency_insitu, axis=1),
            zs=-82,
            zdir="x",
            linewidth=3,
            label="In-situ Cirrus",
            alpha=0.9,
        )
        ax.plot(
            kwargs["heights"],
            np.nanmax(cloud_frequency_anvil, axis=1),
            zs=-82,
            zdir="x",
            linewidth=3,
            label="Anvil Cirrus",
            alpha=0.9,
        )
        ax.plot(
            kwargs["latitudes"],
            np.nanmax(cloud_frequency_insitu, axis=0) + 0.0015,
            zs=-4.8,
            zdir="y",
            linewidth=3,
            alpha=0.9,
            color="black",
        )

        if plot_flag_trop:
            # Plot tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Trop_Height"],
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                zorder=10,
            )

        if plot_flag == 0:
            # Plot 3km below tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Trop_Height"] - 3,
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                linestyle="--",  # This sets the line style to dashed
                zorder=10,
            )
        elif plot_flag == 1:
            # Plot 3km below tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Z300_Height"],
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                linestyle="--",  # This sets the line style to dashed
                zorder=10,
            )
        elif plot_flag == 2:
            # Plot 3km below tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Hgt_200hPa_lower_Trop"],
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                linestyle="--",  # This sets the line style to dashed
                zorder=10,
            )

        # Fill the space from the black line to the coordinate axis with gray
        # Define the vertices for the polygon
        verts = [
            (
                kwargs["latitudes"][i],
                -4.8,
                np.max(cloud_frequency_insitu, axis=0)[i] + 0.0015,
            )
            for i in range(len(kwargs["latitudes"]))
        ]
        # Close the loop for the polygon
        verts += [
            (
                kwargs["latitudes"][-1],
                -4.8,
                0,
            ),  # Last point at the bottom of the axis
            (
                kwargs["latitudes"][0],
                -4.8,
                0,
            ),  # First point at the bottom of the axis
        ]

        # Create a Poly3DCollection object
        poly = Poly3DCollection([verts], facecolors="gray", alpha=0.8)

        # Add the polygon to the plot
        ax.add_collection3d(poly)

        # Apply the colormap to the normalized data to get RGB values
        cmap = plt.get_cmap("RdYlBu_r").copy()
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad("white")

        # Plot the surface
        surf = ax.plot_surface(
            lat_mesh,
            height_mesh,
            cloud_frequency_insitu_surf,
            rcount=500,
            ccount=500,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # facecolors=shaded_rgb,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        # Turn off axis labels and ticks
        ax.set_xlabel("Latitude")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_zticks(zticks)
        ax.set_zlim(zlim)

        ax.set_xlim(-82, 82)

        ax.set_box_aspect([1.53, 1, 2])
        ax.legend()

        # Set the view angle
        ax.view_init(elev=elev, azim=azim, roll=roll)

        ax.set_facecolor("none")
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(
            True,
            color="silver",
            linestyle="-.",
            linewidth=0.3,
            alpha=0.4,
        )

        plt.show()

    def cld_3d_sructure_plot_fig1(
        cloud_frequency,
        zlable,
        zticks=np.arange(0, 70, 20),
        zlim=slice(0, 50),
        elev=110,
        azim=-170,
        roll=-80,
        vmin=0,
        vmax=60,
        # Plot flag to control whether tropopause is plotted
        plot_flag_trop=True,
        # Plot flag to control which heights are plotted
        # 0 means 3km below tropopause, 1 means 300 hPa, 2 means 200 hPa below tropopause
        plot_flag=0,
        **kwargs,
    ):
        cloud_frequency_surf = np.copy(cloud_frequency)
        cloud_frequency_surf[cloud_frequency_surf == 0] = -0.0000001

        # Set up the plot
        fig, ax = plt.subplots(
            subplot_kw=dict(projection="3d"),
            figsize=(11, 9.5),
            dpi=330,
            tight_layout=True,
        )

        lat_mesh, height_mesh = np.meshgrid(
            kwargs["latitudes"], kwargs["heights"]
        )

        if plot_flag_trop:
            # Plot tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Trop_Height"],
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                zorder=10,
            )

        if plot_flag == 0:
            # Plot 3km below tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Trop_Height"] - 3,
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                linestyle="--",  # This sets the line style to dashed
                zorder=10,
            )
        elif plot_flag == 1:
            # Plot 3km below tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Z300_Height"],
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                linestyle="--",  # This sets the line style to dashed
                zorder=10,
            )
        elif plot_flag == 2:
            # Plot 3km below tropopause height
            ax.plot(
                kwargs["latitudes"],
                kwargs["Hgt_200hPa_lower_Trop"],
                zs=0,
                zdir="z",
                linewidth=3,
                alpha=0.8,
                color="black",
                linestyle="--",  # This sets the line style to dashed
                zorder=10,
            )

        ax.plot(
            kwargs["latitudes"],
            np.nanmax(cloud_frequency, axis=0),
            zs=-4.8,
            zdir="y",
            linewidth=3,
            label="In-situ Cirrus",
            alpha=0.9,
            color="black",
        )

        # Fill the space from the black line to the coordinate axis with gray
        # Define the vertices for the polygon
        verts = [
            (
                kwargs["latitudes"][i],
                -4.8,
                np.max(cloud_frequency, axis=0)[i],
            )
            for i in range(len(kwargs["latitudes"]))
        ]
        # Close the loop for the polygon
        verts += [
            (
                kwargs["latitudes"][-1],
                -4.8,
                0,
            ),  # Last point at the bottom of the axis
            (
                kwargs["latitudes"][0],
                -4.8,
                0,
            ),  # First point at the bottom of the axis
        ]

        # Create a Poly3DCollection object
        poly = Poly3DCollection([verts], facecolors="gray", alpha=0.8)

        # Add the polygon to the plot
        ax.add_collection3d(poly)

        # Apply the colormap to the normalized data to get RGB values
        cmap = plt.get_cmap("RdYlBu_r").copy()
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad("white")

        # Plot the surface
        surf = ax.plot_surface(
            lat_mesh,
            height_mesh,
            cloud_frequency_surf,
            rcount=500,
            ccount=500,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # facecolors=shaded_rgb,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        # Add a colorbar
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
        mappable.set_array(cloud_frequency)
        fig.colorbar(mappable, ax=ax, shrink=0.47, aspect=12, label=zlable)

        # Set labels
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Height (km)")
        ax.set_zlabel(zlable)
        ax.set_xlim(-82, 82)
        ax.set_zticks(zticks)
        ax.set_zlim(zlim)

        ax.set_box_aspect([1.53, 1, 2])

        # Set the view angle
        ax.view_init(elev=elev, azim=azim, roll=roll)

        ax.set_facecolor("none")
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(
            True,
            color="silver",
            linestyle="-.",
            linewidth=0.3,
            alpha=0.4,
        )

        plt.show()

    pass



##############################################################################################################
