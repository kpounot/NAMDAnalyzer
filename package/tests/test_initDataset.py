import os

filePath = os.path.dirname(os.path.abspath(__file__))

from NAMDAnalyzer.Dataset import Dataset


def test_initDataset_default(emptyDataset):
    """ First, test if the class is correctly instanciated. """

    assert isinstance(emptyDataset, Dataset),   "No Dataset class instance was created."




def test_initDataset_psf(customDataset):
    """ Test initialization with a .psf file. """

    assert isinstance(customDataset(filePath + '/test_data/ubq_ws.psf'), Dataset)




def test_initDataset_dcd(customDataset):
    """ Test initialization with a .dcd file. """

    assert isinstance(customDataset(filePath + '/test_data/ubq_ws.dcd'), Dataset)




def test_initDataset_vel(customDataset):
    """ Test initialization with a .vel file. """

    assert isinstance(customDataset(filePath + '/test_data/ubq_ws.vel'), Dataset)




def test_initDataset_log(customDataset):
    """ Test initialization with a .log file. """

    assert isinstance(customDataset(filePath + '/test_data/ubq_ws.log'), Dataset)




def test_initDataset_pdb(customDataset):
    """ Test initialization with a .pdb file. """

    assert isinstance(customDataset(filePath + '/test_data/ubq_ws.pdb'), Dataset)




def test_initDataset_full(fullDataset):
    """ Test the initialization of a dataset with all types of file. """

    assert isinstance(fullDataset, Dataset)




def test_importFile_guessType(emptyDataset):
    """ Test the importFile method without file type specification. """

    emptyDataset.importFile(filePath + '/test_data/ubq_ws.psf')

    assert emptyDataset.psfData.atoms.shape == (6682, 9)




def test_importFile_fixedType(emptyDataset):
    """ Test the importFile method with file type specification. """

    emptyDataset.importFile(filePath + '/test_data/ubq_ws.psf', fileType='psf')

    assert emptyDataset.psfData.atoms.shape == (6682, 9)



def test__selection(fullDataset):
    """ Test a standard selection string. """

    sel = fullDataset.selection('name OH2')

    assert sel.size == 1817
