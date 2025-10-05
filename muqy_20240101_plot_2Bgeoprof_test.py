# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2025-09-25 16:52:59

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from muqy_20240101_2Bgeoprof_reader import Reader


def format_lat_lon_ticks(ax, lat_data):
    """
    Format x-axis ticks to show latitude coordinates in degrees.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to format.
    lat_data : np.ndarray
        Array of latitude data (x-axis values).
    """
    # 让matplotlib自动选择tick位置
    ax.locator_params(axis='x', nbins=6)

    # 获取当前的tick位置
    tick_positions = ax.get_xticks()

    # 过滤掉超出数据范围的tick
    valid_ticks = tick_positions[
        (tick_positions >= lat_data.min()) & (tick_positions <= lat_data.max())
    ]

    # 格式化标签
    tick_labels = []
    for lat in valid_ticks:
        if lat >= 0:
            tick_labels.append(f"{abs(lat):.0f}°N")
        else:
            tick_labels.append(f"{abs(lat):.0f}°S")

    ax.set_xticks(valid_ticks)
    ax.set_xticklabels(tick_labels)


def set_map(ax):
    """
    Set up the map on the provided axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to set up the map on.
    """
    # Set up the map
    proj = ccrs.PlateCarree()

    # Set up the coastlines
    ax.coastlines(lw=0.5)

    # Set up the gridlines
    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-90, 91, 30)

    # Set up the x and y tick labels
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Set up the font size
    ax.tick_params("both", labelsize=14)
    ax.set_global()


def draw_track(ax, lon1D, lat1D):
    """
    Draw the satellite track based on latitude and longitude.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the track on.
    lon1D : np.ndarray
        Array of longitudes.
    lat1D : np.ndarray
        Array of latitudes.
    """
    # Draw the track
    ax.plot(lon1D, lat1D, lw=2, color="b", transform=ccrs.Geodetic())

    # Draw the start point
    ax.plot(lon1D[0], lat1D[0], "ro", ms=3, transform=ccrs.PlateCarree())
    # Add the start label
    ax.text(
        lon1D[0] + 5,
        lat1D[0],
        "start",
        color="r",
        fontsize=14,
        transform=ccrs.PlateCarree(),
    )


def draw_cross_section(
    ax,
    lat,
    hgt,
    data,
    colormap="Spectral_r",
    cbar_label="Cloud Fraction (%)",
    cbar_orientation="vertical",
    draw_contour=True,
    draw_contour_micro=False,
    cax=None,
    vmin=None,
    vmax=None,
):
    """
    Draw the time-height cross-section of the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the cross-section on.
    time : np.ndarray
        Array of time data.
    hgt : np.ndarray
        Array of height data.
    data : np.ndarray
        The data to be plotted.
    colormap : str, optional
        The colormap used for the plot. Default is "Spectral_r".
    cbar_label : str, optional
        The label for the colorbar. Default is "Cloud Fraction (%)".
    cbar_orientation : str, optional
        Orientation of the colorbar ("vertical" or "horizontal"). Default is "vertical".
    """
    # Set up the colormap
    cmap = plt.get_cmap(colormap)
    cmap.set_bad("white", 1.0)
    im = ax.pcolormesh(lat, hgt, data.T, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylim(0, 22)
    ax.set_xlim(lat.min(), lat.max())
    # ax.tick_params(labelsize=11)

    format_lat_lon_ticks(ax, lat)
    ax.set_xlabel("Latitude", fontsize=12)
    ax.set_ylabel("Height (km)", fontsize=12)

    # Add contour to draw a solid black line around the plotted subject
    if draw_contour:
        ax.contour(
            lat,
            hgt,
            data.T,
            levels=[0.5],
            colors="black",
            linewidths=2,
        )

    if draw_contour_micro:
        ax.contour(
            lat,
            hgt,
            data.T,
            levels=[0],
            colors="black",
            linewidths=2,
        )

    cbar = plt.colorbar(im, cax=cax, orientation=cbar_orientation)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(cbar_label, fontsize=12)


def draw_cross_section_lidar_radar(ax, lat, hgt, data, uncertainty):
    """
    Draw the time-height cross-section of clouds captured by radar, lidar, or both.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the cross-section on.
    time : np.ndarray
        Array of time data.
    hgt : np.ndarray
        Array of height data.
    data : np.ndarray
        The data to be plotted.
    uncertainty : np.ndarray
        The UncertaintyCF data indicating the source of detection.
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # Create masked arrays for each category
    no_data = np.ma.masked_where(uncertainty != 0, data, copy=True)
    radar_only = np.ma.masked_where(uncertainty != 1, data, copy=True)
    lidar_only = np.ma.masked_where(uncertainty != 2, data, copy=True)
    both = np.ma.masked_where(uncertainty != 3, data, copy=True)

    # Define colors for each category
    color_none = "white"
    color_radar = "blue"
    color_lidar = "green"
    color_both = "orange"

    # Plot each category
    ax.pcolormesh(lat, hgt, no_data.T, color=color_none, shading="nearest")
    ax.pcolormesh(
        lat,
        hgt,
        radar_only.T,
        color=color_radar,
        shading="nearest",
    )
    ax.pcolormesh(
        lat,
        hgt,
        lidar_only.T,
        color=color_lidar,
        shading="nearest",
    )
    ax.pcolormesh(lat, hgt, both.T, color=color_both, shading="nearest")

    ax.set_ylim(0, 20)
    ax.set_xlim(0, lat.max())
    ax.tick_params(labelsize=11)

    format_lat_lon_ticks(ax, lat)
    ax.set_xlabel("Latitude", fontsize=12)
    ax.set_ylabel("Height (km)", fontsize=12)

    # Create a custom color bar
    colors = [color_none, color_radar, color_lidar, color_both]
    labels = ["None", "Radar Only", "Lidar Only", "Both"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="10%", pad=0.5)
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0.5, 1.5, 2.5, 3.5],
    )
    cbar.ax.set_xticklabels(labels)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Cloud Detection Source", fontsize=12)


def draw_elevation(ax, lat, elv):
    """
    Draw the elevation profile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the elevation on.
    lat : np.ndarray
        Array of latitude data.
    elv : np.ndarray
        Array of elevation data.
    """
    ax.fill_between(lat, elv, color="gray")
    format_lat_lon_ticks(ax, lat)  # 为elevation图也添加纬度格式化


def filter_by_time_range(data, time, start, end):
    """
    Filter data based on a specified time range.

    Parameters:
        data (array-like): The data to be filtered.
        time (array-like): The time values corresponding to the data.
        start (float): The start time of the range.
        end (float): The end time of the range.

    Returns:
        tuple: A tuple containing the filtered data and corresponding time values.
    """
    mask = (time >= start) & (time <= end)
    return data[mask], time[mask]


if __name__ == "__main__":
    pass
