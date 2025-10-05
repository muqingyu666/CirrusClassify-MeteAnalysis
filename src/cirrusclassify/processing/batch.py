"""High-level orchestration for the cirrus classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from cirrusclassify.io.cloudsat_reader import CloudSatReader
from cirrusclassify.processing.features import FeatureEngineer


@dataclass
class BatchProcessor:
    config: Dict[str, Any]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BatchProcessor":
        return cls(config=config)

    def run(self) -> None:
        input_dir = Path(self.config["data"].get("cloudsat_dir"))
        files = sorted(input_dir.glob("*.hdf"))
        engineer = FeatureEngineer.from_config(self.config["features"])

        for file_path in files:
            with CloudSatReader.from_file(file_path) as reader:
                geo = reader.read_geo()
                profile = reader.read_sds(self.config["data"]["primary_sds"])  # type: ignore[arg-type]
                features = engineer.transform(geo, profile)
                self._persist_output(file_path, features)

    def _persist_output(self, file_path: Path, features: pd.DataFrame) -> None:
        output_dir = Path(self.config["output"].get("directory", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{file_path.stem}_features.parquet"
        features.to_parquet(target, index=False)
