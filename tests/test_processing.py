from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cirrusclassify.processing.batch as batch_module
from cirrusclassify.processing.batch import BatchProcessor
from cirrusclassify.processing.features import FeatureEngineer


def test_feature_engineer_transform_assigns_regions():
    engineer = FeatureEngineer.from_config({"primary_variable": "Cloud_mask"})

    geo = (
        np.array([0.0, 120.0, -45.0]),
        np.array([0.0, 55.0, -70.0]),
        np.array([100.0, 200.0, 300.0]),
    )
    cloud_mask = np.array([1, 2, 3])

    df = engineer.transform(geo, cloud_mask)

    assert list(df["region"]) == ["tropics", "mid_lat_n", "polar_s"]
    assert df["Cloud_mask"].tolist() == [1.0, 2.0, 3.0]
    assert df["elevation"].tolist() == [100.0, 200.0, 300.0]


def test_batch_processor_run_invokes_reader_and_engineer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sample_file = tmp_path / "granule_20200101.hdf"
    sample_file.write_text("dummy")

    config = {
        "data": {"cloudsat_dir": str(tmp_path), "primary_sds": "Cloud_mask"},
        "features": {"primary_variable": "Cloud_mask"},
        "output": {"directory": str(tmp_path / "outputs")},
    }

    class DummyReader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @classmethod
        def from_file(cls, _path: str | Path):
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read_geo(self):
            return (
                np.array([0.0, 10.0]),
                np.array([0.0, 10.0]),
                np.array([100.0, 200.0]),
            )

        def read_sds(self, _varname: str):
            return np.array([1, 2])

    class DummyEngineer:
        def __init__(self, config):
            self.config = config

        @classmethod
        def from_config(cls, config):
            return cls(config)

        def transform(self, geo, profile):
            lon, lat, elev = geo
            return pd.DataFrame(
                {
                    "longitude": lon,
                    "latitude": lat,
                    "elevation": elev,
                    "mask": profile,
                }
            )

    monkeypatch.setattr(batch_module, "CloudSatReader", DummyReader)
    monkeypatch.setattr(batch_module, "FeatureEngineer", DummyEngineer)

    captured = {}

    def fake_persist(self, file_path: Path, features: pd.DataFrame) -> None:
        captured[file_path.name] = features

    monkeypatch.setattr(BatchProcessor, "_persist_output", fake_persist, raising=False)

    BatchProcessor.from_config(config).run()

    assert "granule_20200101.hdf" in captured
    assert len(captured["granule_20200101.hdf"]) == 2
