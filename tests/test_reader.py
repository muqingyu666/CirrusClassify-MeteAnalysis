import numpy as np
import pytest

from cirrusclassify.io.cloudsat_reader import CloudSatReader


class DummyHDF:
    def __init__(self):
        self.vs = self
        self.sd = self

    # Vdata interface
    def vstart(self):
        return self

    def attach(self, name):
        class _Vdata:
            def __init__(self, name):
                self._name = name

            def __getitem__(self, item):
                if "factor" in self._name:
                    return np.array([[100]])
                if "valid_range" in self._name:
                    return np.array([[0, 1000]])
                return np.array([[1, 2, 3]])

            def detach(self):
                pass

        return _Vdata(name)

    def end(self):
        pass

    # Scientific dataset interface
    def select(self, name):
        class _SDS:
            def __init__(self, name):
                self._name = name

            def __getitem__(self, item):
                return np.array([10, 20, 30])

        return _SDS(name)

    # File level
    def close(self):
        pass


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setattr("pyhdf.HDF.HDF", lambda *_args, **_kwargs: DummyHDF())
    monkeypatch.setattr("pyhdf.SD.SD", lambda *_args, **_kwargs: DummyHDF())

    class _Reader(CloudSatReader):
        def __init__(self):
            self.hdf = DummyHDF()
            self.vs = self.hdf.vstart()
            self.sd = DummyHDF()

    return _Reader()


def test_scale_and_mask(reader):
    data = np.array([0, 50, 100])
    scaled = reader.scale_and_mask(data, "TestVar")
    assert scaled[0] == 0
    assert np.isnan(scaled[-1])
