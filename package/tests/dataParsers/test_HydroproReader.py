from NAMDAnalyzer.dataParsers.HydroproReader import HydroproReader

def test_emptyClassInit(customDataset):
    """ Test initialization of the empty class. """

    data = customDataset('./test_data/ubq_ws.pdb')
    hr = HydroproReader()

    assert isinstance(hr, HydroproReader)




def test_classInit(customDataset):
    """ Test initialization of the class. """

    data = customDataset('./test_data/ubq_ws.pdb')
    hr = HydroproReader(resFile='./test_data/lysozyme_hydropro_result.txt')

    assert hr.params.sedCoeff == 0.6679




def test_readFile(customDataset):
    """ Test initialization of the class. """

    data = customDataset('./test_data/ubq_ws.pdb')
    hr = HydroproReader()
    hr.readFile('./test_data/lysozyme_hydropro_result.txt')

    assert hr.params.sedCoeff == 0.6679
