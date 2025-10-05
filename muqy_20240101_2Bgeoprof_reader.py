# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2024-03-16 10:35:23

from pyhdf import HDF, VS, SD

import numpy as np
import pandas as pd


class Reader:
    """Class for reading CloudSat R05 2B products."""

    def __init__(self, fname):
        """
        Initialize the HDF, Vdata table, and scientific dataset.

        Parameters
        ----------
        fname : str
            File name of the HDF file to read.
        """
        self.hdf = HDF.HDF(fname, HDF.HC.READ)
        self.vs = self.hdf.vstart()
        self.sd = SD.SD(fname, SD.SDC.READ)

    def attach_vdata(self, varname):
        """
        Read all records of a variable from the Vdata table.

        Parameters
        ----------
        varname : str
            Name of the variable to read from Vdata.

        Returns
        -------
        np.ndarray
            Array of the variable data.
        """
        vdata = self.vs.attach(varname)
        data = vdata[:]
        vdata.detach()

        return data

    def scale_and_mask(self, data, varname):
        """
        Scale data based on the variable's factor and mask values outside valid range.

        Parameters
        ----------
        data : np.ndarray
            Data array to be scaled and masked.
        varname : str
            Name of the variable for scaling and masking.

        Returns
        -------
        numpy.ma.array
            Scaled and masked data array.
        """
        factor, (valid_min, valid_max) = (
            self.attach_vdata(f"{varname}.factor")[0][0],
            self.attach_vdata(f"{varname}.valid_range")[0][0],
        )
        invalid = (data < valid_min) | (data > valid_max)

        # Scale the data
        scaled_data = data / factor

        # Replace invalid data points with NaN
        scaled_data[invalid] = np.nan

        return scaled_data

    def read_geo(self, process=True):
        """
        Read longitude, latitude, and elevation data.

        Parameters
        ----------
        process : bool, optional
            Whether to scale and mask elevation data (default is True).

        Returns
        -------
        tuple
            Tuple containing longitude, latitude, and elevation data arrays.
        """
        lon = np.array(self.attach_vdata("Longitude")).ravel()
        lat = np.array(self.attach_vdata("Latitude")).ravel()
        elv = np.array(self.attach_vdata("DEM_elevation")).ravel()

        if process:
            elv = self.scale_and_mask(elv, "DEM_elevation")

        return lon, lat, elv

    def read_time(self, datetime=True):
        """
        Read time data for each data point.

        Parameters
        ----------
        datetime : bool, optional
            If True, returns a DatetimeIndex of all data points (default is True).
            If False, returns a numpy array of seconds elapsed since the first point.

        Returns
        -------
        pandas.DatetimeIndex or np.ndarray
            DatetimeIndex or numpy array of time data.
        """
        profile_seconds = np.array(self.attach_vdata("Profile_time")).ravel()

        if datetime:
            TAI = self.attach_vdata("TAI_start")[0][0]
            start = pd.to_datetime("1993-01-01 00:00:00") + pd.Timedelta(
                seconds=TAI
            )
            times = [start + pd.Timedelta(seconds=s) for s in profile_seconds]
            return pd.DatetimeIndex(times)
        else:
            return profile_seconds

    def read_sds(self, varname, process=True):
        """
        Read data from the scientific dataset.

        Parameters
        ----------
        varname : str
            Name of the variable to read.
        process : bool, optional
            Whether to scale and mask the data (default is True).

        Returns
        -------
        numpy.ma.array or np.ndarray
            Scaled and masked data array or raw data array.
        """
        data = self.sd.select(varname)[:]
        if process:
            data = self.scale_and_mask(data, varname)

        return data

    def read_vdata(self, varname):
        """
        Read the variable data from the Vdata.

        Parameters:
            varname (str): The name of the variable to be read.

        Returns:
            numpy.ndarray: The variable data as a numpy array.
        """
        data = np.array(self.vs.attach(varname)[:]).squeeze()

        return data

    def close(self):
        """Close the HDF file."""
        self.vs.end()
        self.sd.end()
        self.hdf.close()


# Test the Reader class
if __name__ == "__main__":
    pass
