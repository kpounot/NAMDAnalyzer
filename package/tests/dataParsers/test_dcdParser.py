import os 

filePath = os.path.dirname(os.path.abspath(__file__))

def test_appendPDB(emptyDataset):
    """ Test pdb file append function. """

    emptyDataset.appendPDB(filePath + '/../test_data/ubq_ws.pdb')

    assert emptyDataset.nbrFrames == 1

