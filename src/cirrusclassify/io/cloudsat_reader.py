"""Utilities for reading CloudSat R05 Level-2 HDF products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from pyhdf import HDF, SD


@dataclass(slots=True)
class CloudSatReader:
    """Minimal reader around CloudSat R05 2B* HDF files.

    Parameters
    ----------
    fname:
        Path to the granule to open.
    read_mode:
        Optional override of the HDF access mode. Defaults to read-only.
    """

    fname: str
    read_mode: int = HDF.HC.READ

    def __post_init__(self) -> None:
        self.hdf = HDF.HDF(self.fname, self.read_mode)
        self.vs = self.hdf.vstart()
        self.sd = SD.SD(self.fname, SD.SDC.READ)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def attach_vdata(self, varname: str) -> np.ndarray:
        """Return all records for a Vdata table."""

        vdata = self.vs.attach(varname)
        try:
            return np.asarray(vdata[:])
        finally:  # always detach to avoid handle leaks
            vdata.detach()

    def scale_and_mask(self, data: np.ndarray, varname: str) -> np.ndarray:
        """Apply scale factor and valid range masking as encoded in metadata."""

        factor = self.attach_vdata(f"{varname}.factor")[0][0]
        valid_min, valid_max = self.attach_vdata(f"{varname}.valid_range")[0][0]
        scaled = data.astype(float) / factor
        invalid = (scaled < valid_min) | (scaled > valid_max)
        scaled[invalid] = np.nan
        return scaled

    # ------------------------------------------------------------------
    # High-level accessors
    # ------------------------------------------------------------------
    def read_geo(self, process: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return longitude, latitude, and digital elevation arrays."""

        lon = np.asarray(self.attach_vdata("Longitude")).ravel()
        lat = np.asarray(self.attach_vdata("Latitude")).ravel()
        elv = np.asarray(self.attach_vdata("DEM_elevation")).ravel()

        if process:
            elv = self.scale_and_mask(elv, "DEM_elevation")

        return lon, lat, elv

    def read_time(self, as_datetime: bool = True) -> Iterable:
        """Return the observation time per profile."""

        profile_seconds = np.asarray(self.attach_vdata("Profile_time")).ravel()

        if not as_datetime:
            return profile_seconds

        tai_start = self.attach_vdata("TAI_start")[0][0]
        base = pd.to_datetime("1993-01-01 00:00:00") + pd.Timedelta(seconds=tai_start)
        return pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in profile_seconds])

    def read_sds(self, varname: str, process: bool = True) -> np.ndarray:
        """Read a Scientific Data Set (SDS) and optionally scale/mask it."""

        data = np.asarray(self.sd.select(varname)[:])
        if process:
            data = self.scale_and_mask(data, varname)
        return data

    def read_vdata(self, varname: str) -> np.ndarray:
        """Read and squeeze Vdata arrays."""

        return np.asarray(self.vs.attach(varname)[:]).squeeze()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close all open handles."""

        self.vs.end()
        self.sd.end()
        self.hdf.close()

    # ------------------------------------------------------------------
    # Constructors & context manager support
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, read_mode: int | None = None) -> "CloudSatReader":
        if read_mode is None:
            read_mode = HDF.HC.READ
        return cls(str(path), read_mode)

    def __enter__(self) -> "CloudSatReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# Backwards compatibility alias -------------------------------------------------------
Reader = CloudSatReader

__all__ = ["CloudSatReader", "Reader"]
