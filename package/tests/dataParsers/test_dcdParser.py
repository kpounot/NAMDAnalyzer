def test_appendPDB(emptyDataset):
    """ Test pdb file append function. """

    emptyDataset.appendPDB('./test_data/ubq_ws.pdb')

    assert emptyDataset.nbrFrames == 1

