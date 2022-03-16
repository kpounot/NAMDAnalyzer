"""

Classes
^^^^^^^

"""

import os

import numpy as np
import re

from struct import *

try:
    from NAMDAnalyzer.lib.pylibFuncs import py_getDCDCoor, py_getDCDCell
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
          "Please compile it before using it.\n")

from NAMDAnalyzer.selection.selText import SelText
from NAMDAnalyzer.dataParsers.dcdCell import DCDCell
from NAMDAnalyzer.dataParsers.pdbParser import NAMDPDB


class DCDReader:
    """ This class allow user to import a DCD file and provides
        methods to extract the trajectory.

        Data can be obtained by using standard __getitem__ method,
        that is by calling ``dcdData[:,:,:]`` where the first slice
        corresponds to atoms, the second one to the frames,
        and the last one to x, y and z coordinates.

    """
    def __init__(self):
        self.dcdFiles   = []
        self.startPos   = []
        self.initFrame  = []
        self.stopFrame  = []

        self.dcdData    = self
        self.nbrFrames  = 0
        self.timestep   = 0
        self.nbrSteps   = 0
        self.nbrAtoms   = 0
        self.dcdFreq    = []

        self.cell = 0

        self.cellDims = DCDCell(self)

        self.byteorder = '<'

    def __getitem__(self, slices):
        """ Accessor that calls C function to get selected coordinates. """
        atoms   = np.arange(self.nbrAtoms, dtype=int)
        frames  = np.arange(self.nbrFrames, dtype=int)
        dims    = np.arange(3, dtype=int)

        #########################
        # 1D selection - atoms
        #########################
        if isinstance(slices, (slice, range)):
            start = slices.start if slices.start is not None else 0
            stop  = slices.stop if slices.stop is not None else self.nbrAtoms
            step  = slices.step if slices.step is not None else 1
            atoms  = np.arange(start, stop, step)

        elif isinstance(
                slices, 
                (list, np.ndarray, SelText, int, np.int32, np.int64)
        ):
            atoms = np.array([slices]).flatten()

        #########################
        # 2D or 3D selection - atoms and frames (and dimensions)
        #########################
        elif len(slices) == 2 or len(slices) == 3:
            # atoms selection

            if isinstance(slices[0], (slice, range)):
                start = slices[0].start if slices[0].start is not None else 0
                stop  = (slices[0].stop if slices[0].stop is not None
                         else self.nbrAtoms)
                step  = slices[0].step if slices[0].step is not None else 1
                atoms  = np.arange(start, stop, step)

            else:
                atoms = np.array([slices[0]]).flatten()
            
            # slices selection

            if isinstance(slices[1], (slice, range)):
                start = slices[1].start if slices[1].start is not None else 0
                stop  = (slices[1].stop if slices[1].stop is not None
                         else self.nbrFrames)
                step  = slices[1].step if slices[1].step is not None else 1
                frames = np.arange(start, stop, step)
            else:
                frames = np.array([slices[1]]).flatten()


            # possibly axis selection
            if isinstance(slices[1], (int, list, np.ndarray)):
                frames = np.array(
                    [slices[1]]) if isinstance(slices[1], int) else slices[1]

            if len(slices) == 3:
                if isinstance(slices[2], (slice, range)):
                    start = (slices[2].start if slices[2].start
                             is not None else 0)
                    stop  = slices[2].stop if slices[2].stop is not None else 3
                    step  = slices[2].step if slices[2].step is not None else 1

                    dims = np.arange(start, stop, step)

                else:
                    dims = np.array([slices[2]]).flatten()


        elif len(slices) > 3:
            print("Too many dimensions requested, maximum is 3.")
            return

        # Extract coordinates given selected atoms, frames and coordinates
        out = np.zeros((len(atoms), len(frames), len(dims)), dtype='float32')
        tmpOut = None
        for idx, f in enumerate(self.dcdFiles):
            if isinstance(f, str):
                fileFrames = np.arange(self.initFrame[idx],
                                       self.stopFrame[idx], dtype=int)
                fileFrames, id1, id2 = np.intersect1d(
                    fileFrames, frames, return_indices=True)

                tmpOut = np.ascontiguousarray(out[:, id2], dtype='float32')

                try:
                    self._processErrorCode(
                        py_getDCDCoor(
                            bytearray(f, 'utf-8'),
                            id1.astype('int32'),
                            self.nbrAtoms,
                            atoms.astype('int32'),
                            dims.astype('int32'),
                            self.cell,
                            self.startPos[idx].astype('int64'),
                            tmpOut,
                            ord(self.byteorder)
                        )
                    )
                except Exception as e:
                    return

                out[:, id2] = tmpOut

            elif isinstance(f, np.ndarray):
                fileFrames = np.arange(self.initFrame[idx],
                                       self.stopFrame[idx], dtype=int)
                fileFrames, id1, id2 = np.intersect1d(
                    fileFrames, frames, return_indices=True)

                out[:, id2] = f[atoms]

        # Check shape and resize if necessary
        if out.shape[0] > self.nbrAtoms:
            out = out[:self.nbrAtoms]

        if out.shape[1] > self.nbrFrames:
            out = out[:, :self.nbrFrames]

        if out.shape[2] > 3:
            out = out[:, :, :3]

        return np.ascontiguousarray(out)

    def importDCDFile(self, dcdFile):
        """Imports a new file and store the result in *dcdData* attribute.

            If something already exists in *dcdData* attribute,
            it will be deleted.

        """
        self.dcdFiles   = [os.path.abspath(dcdFile)]
        self.dcdFreq    = None
        self.nbrAtoms   = None
        self.nbrFrames  = None
        self.nbrSteps   = None

        self.initFrame = [0]

        with open(dcdFile, 'rb') as f:
            data = f.read(92)

            # Get some simulation parameters (frames, steps and dcd frequency)
            record = unpack('%si4c9if11i' % self.byteorder, data)

            if int(record[0]) != 84:
                self.byteorder = '>'
                record = unpack('%si4c9if11i' % self.byteorder, data)

            self.nbrFrames = int(record[5])
            dcdFreq        = int(record[7])
            self.nbrSteps  = int(record[8])
            self.timestep  = float(record[14] * 0.04888e-12)
            # Whether cell dimensions are given
            self.cell      = bool(record[15])

            if self.nbrFrames > self.nbrSteps:
                self.nbrFrames = self.nbrSteps
                self.nbrSteps = self.nbrFrames * dcdFreq

            self.stopFrame = [self.nbrFrames]

            # Get next record size to skip it (title)
            data = f.read(4)
            titleSize = unpack('i', data)[0]
            data = f.read(titleSize + 4)

            # Get the number of atoms
            data = f.read(12)
            self.nbrAtoms = unpack('%siii' % self.byteorder, data)[1]

            if self.cell:
                # Full size with cell dimensions and 3 coordinates
                recSize = 12 * self.nbrAtoms + 80

                self.startPos = [np.arange(self.nbrFrames, dtype='int64')
                                 * recSize + 112 + titleSize]
            else:
                # Full size with 3 coordinates
                recSize = 12 * self.nbrAtoms + 24

                self.startPos = [np.arange(self.nbrFrames, dtype='int64')
                                 * recSize + 112 + titleSize]

        # Converting dcdFreq to an array of size nbrFrames for handling
        # different dcdFreq during conversion to time
        self.dcdFreq = np.zeros(self.nbrFrames) + dcdFreq

    def appendDCD(self, dcdFile):
        """ Method to append trajectory data to the existing loaded data.

            :arg dcdFile: a single .dcd trajectory file

        """
        try:
            self.dcdData  # Checking if a dcd file has been loaded already
        except AttributeError:
            print("No trajectory file (.dcd) was loaded.\n "
                  "Please load one before using this method.\n")
            return

        tempDatafiles   = self.dcdFiles
        tempdcdFreq     = self.dcdFreq
        tempnbrFrames   = self.nbrFrames
        tempnbrSteps    = self.nbrSteps
        tempCell        = self.cell
        tempStartPos    = self.startPos
        tempInitFrame   = self.initFrame
        tempStopFrame   = self.stopFrame

        self.importDCDFile(dcdFile)

        self.initFrame[0] += tempnbrFrames
        self.stopFrame[0] += tempnbrFrames

        # Append the new data at the end, along the frame axis ('y' axis)
        self.dcdFiles   = tempDatafiles + self.dcdFiles
        self.dcdFreq    = np.append(tempdcdFreq, self.dcdFreq)
        self.nbrSteps   += tempnbrSteps
        self.nbrFrames  += tempnbrFrames
        self.cell       = tempCell * self.cell
        self.startPos   = tempStartPos + self.startPos
        self.initFrame  = tempInitFrame + self.initFrame
        self.stopFrame  = tempStopFrame + self.stopFrame

    def _processErrorCode(self, error_code):
        """ Used to process return value of py_getDCDCoor function. """
        if error_code == 0:
            return
        if error_code == -1:
            raise IOError("Error while opening the file. "
                          "Please check file path or access permissions.\n")
        if error_code == -2:
            raise IndexError("Out of range index. "
                             "Please check again requested slices.\n")
        if error_code == -3:
            raise IOError("Error while reading the file.\n")
