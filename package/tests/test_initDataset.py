import pytest

from ..NAMDAnalyzer.Dataset import Dataset


def test_initDataset_default():
    data = Dataset()

    assert isinstance(data, Dataset)
    assert data.stride == 1
