def test_getCellArray(fullDataset):
    """ Test cell dimensions accessor. """

    assert fullDataset.cellDims[:].shape == (10, 3)
    assert fullDataset.cellDims[0,2].shape == (1, 1)
