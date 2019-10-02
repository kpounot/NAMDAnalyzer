"""

Classes
^^^^^^^

"""

import re

import numpy as np

from NAMDAnalyzer.selection.selParser import SelParser


class SelText():
    """ This class provides methods to easily access various attributes of a given selection.

        That is, selected indices, coordinates, residues, segment names,... can be accessed 
        from the given :class:`Dataset` class using appropriate methods.

        :arg dataset: a :class:`Dataset` class instance containing psf and dcd data
        :arg selText: a selection string (default 'all')

        If no frame is selected, all of them will be returned with :py:func:`coordinates` method.

        .. warning:: 
            
            The behavior is not well defined when the multiple frames are used: ``'...within...frame 0:50:2'``.
            Especially, the *_indices* attribute becomes a list of array, therefore iteration or
            slicing over the class itself won't work correctly.

    """

    def __init__(self, dataset, selT='all'):

        self.dataset = dataset
        self.selT    = selT

        
        tempSel = SelParser(self.dataset, self.selT)

        self._indices = tempSel.selection
        self.frames   = tempSel.frame

        self.shape = self._indices.shape if isinstance(self._indices, np.ndarray) else len(self._indices)
        self.size  = self._indices.size if isinstance(self._indices, np.ndarray) else len(self._indices)

        self.iterIdx = 0



    def __getitem__(self, index):
        """ Makes sel iterable over _indices. 

            This calls *_indices* method directly. Such that everything is managed by numpy.ndarray object.

        """

        return self._indices[index]



    def __len__(self):
        """ Returns length of _indices() array. """

        return self._indices.size



    def astype(self, t):
        """ Redefines numpy function for SelText type. """

        return self._indices.astype(t)



    def getIndices(self):
        """ Returns indices corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,0]



    def getSegName(self):
        """ Returns residues corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,1]


    def getUniqueSegName(self):
        """ Returns an array of str with each segment name in selection apparing only once. """

        segList = np.unique( self.dataset.psfData.atoms[ self._indices ][:,1] )

        return segList


    def getResidues(self):
        """ Returns residues corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,2]


    def getUniqueResidues(self):
        """ Returns an array of str with each residue number in selection apparing only once. """

        resList = np.unique( self.dataset.psfData.atoms[ self._indices ][:,2].astype(int) )

        return resList.astype(str)


    def getResName(self):
        """ Returns residues names corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,3]


    def getUniqueResName(self):
        """ Returns an array of str with each residue name in selection apparing only once. """

        resList = np.unique( self.dataset.psfData.atoms[ self._indices ][:,3] )

        return resList



    def getAtom(self):
        """ Returns atom corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,5]


    def getUniqueAtom(self):
        """ Returns an array of str with each atom in selection apparing only once. """

        atomList = np.unique( self.dataset.psfData.atoms[ self._indices ][:,5] )

        return atomList


    def getName(self):
        """ Returns atom name corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,4]


    def getUniqueAtomName(self):
        """ Returns an array of str with each atom name in selection apparing only once. """

        atomList = np.unique( self.dataset.psfData.atoms[ self._indices ][:,4] )

        return atomList



    def getCharges(self):
        """ Returns charges corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,6].astype(float)



    def getMasses(self):
        """ Returns masses corresponding to each selected atoms in psf file. """

        return self.dataset.psfData.atoms[ self._indices ][:,7].astype(float)




    def coordinates(self, frames=None):
        """ Returns trajectory coordinates for selected frames. 
            
            If ``frames`` argument is None, use default frame from selection text. Else, a integer,
            a range, a slice, a list or a numpy.ndarray can be used.

            The returned array is always 3D.
            In the case of one frame, the shape is (number of atoms, 1, 3).

            For multiple frames, it depends on the kind of selection. If ``'within'`` keyword was used,
            the selection size might change from frame to frame, then a list of 3D coordinates arrays
            is returned.
            Else, for 'static selection', a 3D array of shape (number of atoms, number of frames, 3)
            is returned.

        """

        if frames is None:
            frames = self.frames

        if isinstance(self._indices, list):
            return [self.dataset.dcdData[sel, frames] for i, sel in enumerate(self._indices)]
        else:
            return self.dataset.dcdData[self._indices, frames]


        



