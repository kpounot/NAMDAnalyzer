"""

Classes
^^^^^^^

"""

import re
import os

import numpy as np

from NAMDAnalyzer.selection.selParser import SelParser
from NAMDAnalyzer.helpersFunctions.objectConverters import fromSliceToArange


class SelText():
    """ This class provides methods to easily access various attributes
        of a given selection.

        That is, selected indices, coordinates, residues,
        segment names,... can be accessed from the given
        :class:`Dataset` class using appropriate methods.

        :arg dataset: a :class:`Dataset` class instance containing
                      psf and dcd data
        :arg selText: a selection string (default 'all'), can also
                      be 1D array of indices

        If no frame is selected, all of them will be returned with
        :py:func:`coordinates` method.

        .. warning::

            When multiple frames are used: ``'...within...frame 0:50:2'``,
            the *_indices* attribute becomes a list of array, corresponding
            to the selection for each frame. Also, the frames attributes stores
            the selected frames, such that the :py:method:`coordinates`
            can only returns the frames that are both in :py:attr:`frames`
            class attribute and in the user-requested frames.

    """
    def __init__(self, dataset, selT='all'):
        self.dataset = dataset

        if isinstance(selT, str):
            self.selT = selT
            tempSel = SelParser(self.dataset, self.selT)

            self._indices = tempSel.selection
            self.frames   = tempSel.frame
            if isinstance(self.frames, slice):
                self.frames = fromSliceToArange(self.frames,
                                                self.dataset.nbrFrames)
        elif isinstance(
                selT, (range, list, np.ndarray, int, np.int32, np.int64)
        ):
            selT = np.array(selT).flatten().astype(str)
            self.selT = "index " + " ".join(selT)
            tempSel = SelParser(self.dataset, self.selT)
            self._indices = tempSel.selection
            self.frames   = tempSel.frame
            if isinstance(self.frames, slice):
                self.frames = fromSliceToArange(self.frames,
                                                self.dataset.nbrFrames)
        else:
            self.selT     = ''
            self._indices = selT
            self.frames   = 0

        self.shape = (self._indices.shape
                      if isinstance(self._indices, np.ndarray)
                      else len(self._indices))
        self.size  = (self._indices.size
                      if isinstance(self._indices, np.ndarray)
                      else len(self._indices))

        self.iterIdx = 0

    def __getitem__(self, index):
        """Makes sel iterable over _indices.

        For standard slices or integers, this calls *_indices* method directly. 
        Such that everything is managed by numpy.ndarray object.

        If index if a string or another SelText instance, it selects the 
        indices from self that match with the other selection.

        """
        if isinstance(index, (str, SelText)):
            return self.getSubSelection(index, returnOriId=True)[1]

        return self._indices[index]

    def __len__(self):
        """ Returns length of _indices() array. """
        return self._indices.size

    def __add__(self, addSel):
        """ Allows to add two different selections to get and concatenated one.

            The first selection can contain one or multiple frames.
            That is, *_indices* attribute can be a list of index lists.
            But the second selection should contain a single frame.

        """
        tmp = SelText(self.dataset)

        if isinstance(self._indices, list):
            frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
            tmp._indices = [np.sort(np.concatenate(
                            (self._indices[i], addSel._indices)))
                            for i, val in enumerate(frames)]
        else:
            tmp._indices = np.sort(
                np.concatenate((self._indices, addSel._indices)))

        tmp.selT = self.selT + ' + ' + addSel.selT

        tmp.shape = (tmp._indices.shape if isinstance(tmp._indices, np.ndarray)
                     else len(tmp._indices))
        tmp.size  = (tmp._indices.size if isinstance(tmp._indices, np.ndarray)
                     else len(tmp._indices))
        tmp.iterIdx = 0

        return tmp

    def append(self, indices):
        """ Allows to directly append a list of indices to the selection. """
        if isinstance(self._indices, list):
            frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
            self._indices = [np.sort(np.concatenate(
                             (self._indices[i], indices)))
                             for i, val in enumerate(frames)]
        else:
            self._indices = np.sort(np.concatenate((self._indices, indices)))

        self.shape = (self._indices.shape
                      if isinstance(self._indices, np.ndarray)
                      else len(self._indices))
        self.size  = (self._indices.size
                      if isinstance(self._indices, np.ndarray)
                      else len(self._indices))
        self.iterIdx = 0

    def getSubSelection(self, sel, returnOriId=False):
        """ Performs a sub-selection using given selection within already
            selected indices. Basically, atoms that are both in initial
            selection and *sel* corresponding one are returned.

            :arg sel:           sub-selection to use
            :arg returnOriId:   if True, return also indices in the range
                                of the first, original selection

        """
        tmp = SelText(self.dataset)

        if isinstance(sel, str):
            tempSel = SelParser(self.dataset, sel)
            if isinstance(self._indices, list):
                frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
                res = [np.intersect1d(self._indices[i],
                                      tempSel.selection,
                                      return_indices=True)
                       for i, frame in enumerate(frames)]
                tmp._indices = [val[0] for val in res]
            else:
                res = np.intersect1d(self._indices,
                                     tempSel.selection,
                                     return_indices=True)
                tmp._indices = res[0]

            tmp.selT = sel + " and " + tmp.selT

        elif isinstance(sel, SelText):
            if isinstance(self._indices, list):
                frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
                res = [np.intersect1d(self._indices[i],
                                      sel._indices,
                                      return_indices=True)
                       for i, frame in enumerate(frames)]
                tmp._indices = [val[0] for val in res]
            else:
                res = np.intersect1d(self._indices,
                                     sel._indices,
                                     return_indices=True)
                tmp._indices = res[0]

            tmp.selT = sel.selT + " and " + tmp.selT

        else:
            if isinstance(self._indices, list):
                frames = fromSliceToArange(self.frames, self.dataset.nbrFrames)
                res = [np.intersect1d(self._indices[i],
                                      sel,
                                      return_indices=True)
                       for i, frame in enumerate(frames)]
                tmp._indices = [val[0] for val in res]
            else:
                res = np.intersect1d(self._indices, sel, return_indices=True)
                tmp._indices = res[0]

            tmp.selT = sel

        tmp.shape = (tmp._indices.shape
                     if isinstance(tmp._indices, np.ndarray)
                     else len(tmp._indices))
        tmp.size  = (tmp._indices.size
                     if isinstance(tmp._indices, np.ndarray)
                     else len(tmp._indices))
        tmp.iterIdx = 0

        if isinstance(self._indices, list):
            oriId = [val[1] for val in res]
        else:
            oriId = res[1]

        if returnOriId:
            return tmp, oriId
        else:
            return tmp

    def astype(self, t):
        """ Redefines numpy function for SelText type. """
        return self._indices.astype(t)

    def getIndices(self):
        """ Returns indices corresponding to each selected atoms in psf file.

        """
        return self.dataset.psfData.atoms[self._indices][:, 0]

    def getSegName(self):
        """ Returns residues corresponding to each selected atoms in psf file.

        """
        return self.dataset.psfData.atoms[self._indices][:, 1]

    def getUniqueSegName(self):
        """ Returns an array of str with each segment name
            in selection apparing only once.

        """
        segList = np.unique(self.dataset.psfData.atoms[self._indices][:, 1])
        return segList

    def getResidues(self):
        """ Returns residues corresponding to each
            selected atoms in psf file.

        """
        return self.dataset.psfData.atoms[self._indices][:, 2]

    def getUniqueResidues(self):
        """ Returns an array of str with each residue number in
            selection apparing only once.

        """
        resList = np.unique(
            self.dataset.psfData.atoms[self._indices][:, 2])

        return resList.astype(str)

    def getResName(self):
        """ Returns residues names corresponding to each
            selected atoms in psf file.

        """
        return self.dataset.psfData.atoms[self._indices][:, 3]

    def getUniqueResName(self):
        """ Returns an array of str with each residue name in
            selection apparing only once.

        """
        resList = np.unique(
            self.dataset.psfData.atoms[self._indices][:, 3])

        return resList

    def getAtom(self):
        """ Returns atom corresponding to each selected atoms in psf file. """
        return self.dataset.psfData.atoms[self._indices][:, 5]

    def getUniqueAtom(self):
        """ Returns an array of str with each atom in
            selection apparing only once.

        """
        atomList = np.unique(self.dataset.psfData.atoms[self._indices][:, 5])
        return atomList

    def getName(self):
        """ Returns atom name corresponding to each
            selected atoms in psf file.

        """
        names = self.dataset.psfData.atoms[self._indices][:, 4]
        names = np.array([name if len(name) == 4
                          else ' ' + name for name in names])
        return names

    def getUniqueName(self):
        """ Returns an array of str with each atom name in
            selection apparing only once.

        """
        atomList = np.unique(self.dataset.psfData.atoms[self._indices][:, 4])
        return atomList

    def getCharges(self):
        """ Returns charges corresponding to each
            selected atoms in psf file.

        """
        return self.dataset.psfData.atoms[self._indices][:, 6].astype(float)

    def getMasses(self):
        """ Returns masses corresponding to each selected atom in psf file. """
        return self.dataset.psfData.atoms[self._indices][:, 7].astype(float)

    def set_nScatLength_inc(self, val):
        """ Allows to change the value of the neutron incoherent scattering
            length for the selected atoms.

        """
        self.dataset.psfData.nScatLength_inc[self._indices] = val

    def set_nScatLength_coh(self, val):
        """ Allows to change the value of the neutron coherent scattering
            length for the selected atoms.

        """
        self.dataset.psfData.nScatLength_coh[self._indices] = val

    def coordinates(self, frames=None):
        """ Returns trajectory coordinates for selected frames.

            If ``frames`` argument is None, use default frame
            from selection text. Else, a integer, a range, a slice, a list
            or a numpy.ndarray can be used.

            The returned array is always 3D.
            In the case of one frame, the shape is (number of atoms, 1, 3).

            For multiple frames, it depends on the kind of selection.
            If ``'within'`` keyword was used, the selection size might change
            from frame to frame, then a list of 3D coordinates arrays
            is returned. Else, for 'static selection', a 3D array of shape
            (number of atoms, number of frames, 3) is returned.

        """
        if frames is None:
            frames = self.frames

        elif isinstance(frames, slice):
            frames = fromSliceToArange(frames, self.dataset.nbrFrames)

        elif isinstance(frames, range):
            frames = np.arange(frames.start, frames.stop, frames.step)

        else:
            frames = frames

        if isinstance(self._indices, list):
            frames, selId, usrId = np.intersect1d(
                self.frames, frames, return_indices=True)
            coor = [self.dataset.dcdData[self._indices[selId[i]], int(frame)]
                    for i, frame in enumerate(frames)]

            return coor

        else:
            return self.dataset.dcdData[self._indices, frames]

    def writePDB(self, fileName=None, frame=0, coor=None):
        """ This provides a way to write a simple .pdb file containing
            selected atoms.

            :arg fileName: file name to be used. If None (default), the
                           loaded .psf file name is used.
            :arg frame:    frame to be used
            :arg coor:     if not None, it will override the *frame* argument
                           and directly use the given coordinates instead.

        """
        if fileName is None:
            fileName = self.dataset.psfFile[:-4]

        if coor is None:
            coor = self.coordinates(frame).squeeze()

        with open(fileName + '.pdb', 'w') as f:

            names   = self.getName()
            resName = self.getResName()
            resID   = self.getResidues()
            segName = self.getSegName()

            cD = self.dataset.cellDims[frame].squeeze()

            f.write('CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n'
                    % (cD[0], cD[1], cD[2], 90.00, 90.00, 90.00))


            for idx, val in enumerate(coor):
                f.write('ATOM  %5s %-4s%1s%-4s%1s%4s%1s   '
                        '%8.3f%8.3f%8.3f%6.2f%6.2f      %3s     \n'
                        % (str(idx + 1) if (idx + 1) < 99999 else '*****',
                           names[idx], ' ', resName[idx],
                           segName[idx][0], resID[idx], ' ',
                           val[0], val[1], val[2],
                           0.00, 0.00, segName[idx]))

            f.write('END\n')
