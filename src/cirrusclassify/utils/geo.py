"""Geospatial helper utilities."""

from __future__ import annotations

import numpy as np

REGIONS = {
    "tropics": (-23.5, 23.5),
    "mid_lat_n": (23.5, 66.5),
    "mid_lat_s": (-66.5, -23.5),
    "polar_n": (66.5, 90.0),
    "polar_s": (-90.0, -66.5),
}


def compute_region(latitude: np.ndarray, _longitude: np.ndarray) -> np.ndarray:
    labels = np.full(latitude.shape, "unknown", dtype=object)
    for name, (lower, upper) in REGIONS.items():
        mask = (latitude >= lower) & (latitude <= upper)
        labels = np.where(mask, name, labels)
    return labels
