import os
import pytest

from NAMDAnalyzer.Dataset import Dataset

filePath = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def emptyDataset():
    return Dataset()


@pytest.fixture
def fullDataset(scope='package'):
    return Dataset( filePath + '/test_data/ubq_ws.psf', 
                    filePath + '/test_data/ubq_ws.dcd',
                    filePath + '/test_data/ubq_ws.pdb',
                    filePath + '/test_data/ubq_ws.log',
                    filePath + '/test_data/ubq_ws.vel' )


@pytest.fixture
def customDataset():
    def _make_customDataset(*dataFiles):
        return Dataset(*dataFiles)

    return _make_customDataset


