"""

Classes
^^^^^^^

"""

import re, os

import numpy as np

from NAMDAnalyzer.selection.selParser import SelParser
from NAMDAnalyzer.helpersFunctions.objectConverters import fromSliceToArange


class SelText():
    """ This class provides methods to easily access various attributes of a given selection.

        That is, selected indices, coordinates, residues, segment names,... can be accessed 
        from the given :class:`Dataset` class using appropriate methods.

        :arg dataset: a :class:`Dataset` class instance containing psf and dcd data
        :arg selText: a selection string (default 'all'), can also be 1D array of indices

        If no frame is selected, all of them will be returned with :py:func:`coordinates` method.

        .. warning:: 
            
            The behavior is not well defined when multiple frames are used: ``'...within...frame 0:50:2'``.
            Especially, the *_indices* attribute becomes a list of array, therefore iteration or
            slicing over the class itself won't work correctly.
            In case of selection over dcdData, use d.dcdData[mySelection[0],0] to select coordinates 
            corresponding to first frame.

    """

    def __init__(self, dataset, selT='all'):

        self.dataset = dataset

        if isinstance(selT, str):
            self.selT    = selT
            tempSel = SelParser(self.dataset, self.selT)

            self._indices = tempSel.selection
            self.frames   = tempSel.frame
        else:
            self.selT     = ''
            self._indices = selT
            self.frames   = 0

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




    def __add__(self, addSel):
        """ Allows to add two different selections to get and concatenated one. 
        
            The first selection can contain one or multiple frames. That is, *_indices* attribute can be
            a list of index lists. But the second selection should contain a single frame.

        """

        tmp = SelText(self.dataset) 

        if isinstance(self._indices, list):
            frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
            tmp._indices = [np.sort( np.concatenate( (self._indices[i], addSel._indices) ) ) 
                            for i, val in enumerate(frames)]

        else:
            tmp._indices = np.sort( np.concatenate( (self._indices, addSel._indices) ) ) 


        tmp.selT = self.selT + ' + ' + addSel.selT

        tmp.shape = tmp._indices.shape if isinstance(tmp._indices, np.ndarray) else len(tmp._indices)
        tmp.size  = tmp._indices.size if isinstance(tmp._indices, np.ndarray) else len(tmp._indices)

        tmp.iterIdx = 0

        return tmp




    def append(self, indices):
        """ Allows to directly append a list of indices to the selection. """


        if isinstance(self._indices, list):
            frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
            self._indices = [np.sort( np.concatenate( (self._indices[i], indices) ) ) 
                            for i, val in enumerate(frames)]

        else:
            self._indices = np.sort( np.concatenate( (self._indices, indices) ) ) 


        self.shape = self._indices.shape if isinstance(self._indices, np.ndarray) else len(self._indices)
        self.size  = self._indices.size if isinstance(self._indices, np.ndarray) else len(self._indices)

        self.iterIdx = 0






    def getSubSelection(self, sel, returnOriId=False):
        """ Performs a sub-selection using given selection within already selected indices. 
            Basically, atoms that are both in initial selection and *sel* corresponding one 
            are returned. 

            :arg sel:           sub-selection to use
            :arg returnOriId:   if True, return also indices in the range of the total number of atoms
                                in simulation
            
        """

        tmp = SelText(self.dataset) 

        if isinstance(sel, str):
            tempSel      = SelParser(self.dataset, sel)
            if isinstance(self._indices, list):
                frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
                res = [np.intersect1d( self._indices[i], tempSel.selection, return_indices=True )
                        for i, frame in enumerate(frames)]
                tmp._indices = [val[1] for val in res] 
            else:
                res = np.intersect1d( self._indices, tempSel.selection, return_indices=True )
                tmp._indices = res[1] 

            tmp.selT     = sel + " and " + tmp.selT 


        elif isinstance(sel, SelText):
            if isinstance(self._indices, list):
                frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
                res = [np.intersect1d( self._indices[i], sel._indices, return_indices=True )
                        for i, frame in enumerate(frames)]
                tmp._indices = [val[1] for val in res] 
            else:
                res = np.intersect1d( self._indices, sel._indices, return_indices=True )
                tmp._indices = res[1] 

            tmp.selT     = sel.selT + " and " + tmp.selT 


        else:
            if isinstance(self._indices, list):
                frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
                res = [np.intersect1d( self._indices[i], sel, return_indices=True )
                        for i, frame in enumerate(frames)]
                tmp._indices = [val[1] for val in res] 
            else:
                res = np.intersect1d( self._indices, sel, return_indices=True )
                tmp._indices = res[1] 

            tmp.selT     = sel  


        tmp.shape = tmp._indices.shape if isinstance(tmp._indices, np.ndarray) else len(tmp._indices)
        tmp.size  = tmp._indices.size if isinstance(tmp._indices, np.ndarray) else len(tmp._indices)

        tmp.iterIdx = 0


        if isinstance(self._indices, list):
            oriId = [val[0] for val in res]
        else:
            oriId = res[0]
    


        if returnOriId:
            return tmp, oriId
        else:
            return tmp







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

        names = self.dataset.psfData.atoms[ self._indices ][:,4]

        names = np.array( [name if len(name) == 4  else ' ' + name for name in names] )

        return names


    def getUniqueName(self):
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

        if isinstance(frames, slice):
            frames = fromSliceToArange(frames, self.dataset.nbrFrames)
            return [self.dataset.dcdData[sel, frames[i]] for i, sel in enumerate(self._indices)]

        if isinstance(self._indices, list):
            return [self.dataset.dcdData[sel, frames] for i, sel in enumerate(self._indices)]
        else:
            return self.dataset.dcdData[self._indices, frames]



    def writePDB(self, fileName=None, frame=0, coor=None):
        """ This provides a way to write a simple .pdb file containing selected atoms.

            :arg fileName: file name to be used. If None (default), the loaded .psf file name is used.
            :arg frame:    frame to be used
            :arg coor:     if not None, it will override the *frame* argument and directly the given
                           coordinates instead. 

        """
            
        if fileName is None:
            fileName = self.dataset.psfFile[:-4]

        if coor is None:
            coor = self.coordinates(frame).squeeze()


        with open(fileName + '.pdb', 'w') as f:

            names   = self.getName()
            resName = self.getResName()
            resID   = self.getResidues().astype(int)
            segName = self.getSegName()

            cD = self.dataset.cellDims[frame].squeeze()

            f.write('CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n' 
                        % (cD[0], cD[1], cD[2], 90.00, 90.00, 90.00) )


            for idx, val in enumerate(coor):
                f.write('ATOM  %5i %-4s%1s%-4s%1s%4i%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %3s\n'
                        % ( idx+1,  
                            names[idx], ' ', resName[idx], 
                            segName[idx][0], resID[idx], ' ',
                            val[0], val[1], val[2],
                            0.00, 0.00, segName[idx] ) )

            f.write('END\n')
                            
 



