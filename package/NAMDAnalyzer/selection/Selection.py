import numpy as np


from NAMDAnalyzer.selection.selParser import SelParser


class Selection():
    """ This class provides methods to easily access various attributes of a given selection.

        That is, selected indices, coordinates, residues, segment names,... can be accessed 
        from the given :class:`Dataset` class using appropriate methods.

        :arg dataset: a :class:`Dataset` class instance containing psf and dcd data
        :arg selText: a selection string (default 'all')

        Default frame is the last one.

    """

    def __init__(self, dataset, selT='all'):

        self.dataset = dataset
        self.selT = selT



    def __repr__(self):
        """ Redefines __repr__ to directly get indices with class name as for standard numpy array. 
            
            This way, coordinates can also be selected using ``d.dcdData[sel]`` with d
            being a :class:`Dataset` class instance, and sel a :class:`Selection` instance.

        """

        return self._indices().__repr__()



    def _indices(self):
        """ Returns the indices corresponding to the selection string. """

        return SelParser(self.dataset, self.selT).selection





    def indices(self):
        """ Returns indices corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices() ][:,0]



    def getSegName(self):
        """ Returns residues corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices() ][:,1]


    def getUniqueSegName(self):
        """ Returns an array of str with each segment name in selection apparing only once. """

        segList = np.unique( self.dataset.psfData.atoms[ self._indices() ][:,1] )

        return segList


    def getResidues(self):
        """ Returns residues corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices() ][:,2]


    def getUniqueResidues(self):
        """ Returns an array of str with each residue number in selection apparing only once. """

        resList = np.unique( self.dataset.psfData.atoms[ self._indices() ][:,2].astype(int) )

        return resList.astype(str)


    def getResName(self):
        """ Returns residues names corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices() ][:,3]


    def getUniqueResName(self):
        """ Returns an array of str with each residue name in selection apparing only once. """

        resList = np.unique( self.dataset.psfData.atoms[ self._indices() ][:,3] )

        return resList


