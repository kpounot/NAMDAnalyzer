def test_getTrajArray(fullDataset):
    """ Test trajectory coordinates accessor. """

    assert fullDataset.dcdData[:].shape == (6682, 10, 3)
    assert fullDataset.dcdData[:,3:5,2].shape == (6682, 2, 1)




def test_importDCD(emptyDataset):
    """ Test the importDCDFile method. """

    emptyDataset.importDCDFile('./test_data/ubq_ws.dcd')

    assert emptyDataset.nbrFrames == 10




def test_appendDCD(fullDataset):
    """ Test the appendDCD method. """

    fullDataset.appendDCD('./test_data/ubq_ws.dcd')

    assert fullDataset.nbrFrames == 20




