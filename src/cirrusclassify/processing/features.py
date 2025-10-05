"""Feature engineering utilities for cirrus classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from cirrusclassify.utils.geo import compute_region


@dataclass
class FeatureEngineer:
    config: Dict[str, Any]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureEngineer":
        return cls(config=config)

    def transform(self, geo: tuple[np.ndarray, np.ndarray, np.ndarray], profile: np.ndarray) -> pd.DataFrame:
        lon, lat, elevation = geo
        region = compute_region(lat, lon)
        return pd.DataFrame(
            {
                "longitude": lon,
                "latitude": lat,
                "region": region,
                self.config["primary_variable"]: profile.astype(float).ravel(),
                "elevation": elevation,
            }
        )
