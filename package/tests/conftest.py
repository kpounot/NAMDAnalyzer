import pytest

from NAMDAnalyzer.Dataset import Dataset


@pytest.fixture
def emptyDataset():
    return Dataset()


@pytest.fixture
def fullDataset(scope='package'):
    return Dataset( './test_data/ubq_ws.psf', 
                    './test_data/ubq_ws.dcd',
                    './test_data/ubq_ws.pdb',
                    './test_data/ubq_ws.log',
                    './test_data/ubq_ws.vel' )


@pytest.fixture
def customDataset():
    def _make_customDataset(*dataFiles):
        return Dataset(*dataFiles)

    return _make_customDataset


