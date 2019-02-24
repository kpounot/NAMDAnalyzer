import numpy as np
import re

from scipy.io import FortranFile
from struct import *

class DCDReader:
    def __init__(self, stride=1):

        self.dcdData    = None
        self.nbrFrames  = None
        self.timestep   = None
        self.nbrSteps   = None
        self.nbrAtoms   = None
        self.dcdFreq    = None
        self.stride     = stride


    def importDCDFile(self, dcdFile):
        """ Imports a new file and store the result in self.dcdData.
            If something already exists in self.dcdData, it will be deleted. """

        self.dcdFile = dcdFile

        self.dcdData = None #_Free memory in case data were already loaded

        print(self.dcdFile)

        with open(self.dcdFile, 'rb') as f:
            data = f.read(92)

            #_Get some simulation parameters (frames, steps and dcd frequency)
            record = unpack('i4c9if11i', data)

            self.nbrFrames  = record[5]
            dcdFreq         = record[7]
            self.nbrSteps   = record[8]
            self.timestep   = record[14] * 0.04888e-12 
            self.cell       = record[15] #_Whether cell dimensions are given


            #_Get next record size to skip it (title)
            data = f.read(4)
            titleSize = unpack('i', data)[0]
            data = f.read(titleSize+4)

            #_Get the number of atoms
            data = f.read(12)
            self.nbrAtoms = unpack('iii', data)[1]


            #_Allocating memory to make the process much faster
            self.dcdData = np.zeros( (self.nbrAtoms, 3 * int(np.ceil(self.nbrFrames / self.stride)) ), 
                                                                                        dtype=np.float32 )

        
            if self.cell:

                recSize         = 4 * self.nbrAtoms + 8
                self.cellDims   = np.zeros( (3, int(np.ceil(self.nbrFrames / self.stride)) ) )

                for frame in range(self.nbrFrames):
                    data = f.read(56+3*recSize)
                    if (frame % self.stride) == 0:
                        frame = int(frame / self.stride)
                        self.cellDims[:,frame]    = np.array( unpack('6d', data[4:52]) )[[0,2,5]]
                        self.dcdData[:,3*frame]   = unpack('i%ifi' % self.nbrAtoms, 
                                                                    data[56:56+recSize])[1:-1]
                        self.dcdData[:,3*frame+1] = unpack('i%ifi' % self.nbrAtoms, 
                                                                    data[56+recSize:56+2*recSize])[1:-1]
                        self.dcdData[:,3*frame+2] = unpack('i%ifi' % self.nbrAtoms, 
                                                                    data[56+2*recSize:56+3*recSize])[1:-1]
                        

            else:

                recSize = 3 * self.nbrAtoms + 8

                for frame in range(self.nbrFrames):
                    data = f.read(56+3*recSize)
                    if (frame % self.stride) == 0:
                        frame = int(frame / self.stride) 
                        self.dcdData[:,3*frame]   = unpack('i%ifi' % self.nbrAtoms, data[:recSize])[1:-1]
                        self.dcdData[:,3*frame+1] = unpack('i%ifi' % self.nbrAtoms, 
                                                                            data[recSize:2*recSize])[1:-1]
                        self.dcdData[:,3*frame+2] = unpack('i%ifi' % self.nbrAtoms, 
                                                                            data[2*recSize:3*recSize])[1:-1]




        #_The dataset is reshaped so that we have atom index in axis 0, frame number in axis 1, 
        #_and (x, y, z) coordinates in axis 2
        self.dcdData = self.dcdData.reshape(self.dcdData.shape[0], 
                                            int(np.ceil(self.nbrFrames / self.stride)), 
                                            3)


        #_Converting dcdFreq to an array of size nbrFrames for handling different dcdFreq 
        #_during conversion to time
        self.dcdFreq = np.zeros(self.dcdData.shape[1]) + dcdFreq * self.stride

        
        #_Set number of frames to the right value, taking stride parameter into account
        self.nbrFrames = int(np.ceil(self.nbrFrames / self.stride)) 


        if self.stride == 1: #_Removes the additional entry at the end when stride is equal to one.
            self.dcdData    = self.dcdData[:,:-1]
            self.dcdFreq    = self.dcdFreq[:-1]
            self.nbrFrames  = self.nbrFrames - 1
            self.cellDims   = self.cellDims[:,:-1]


    def appendDCD(self, dcdFile):
        """ Method to append trajectory data to the existing loaded data.
            The .dcd file is opened using dcdParser class, then data are append to the existing
            dcdData instance. 

            Input:  a single .dcd trajectory file """
        
        try:
            self.dcdData #_Checking if a dcd file has been loaded already, print an error message if not.
        except AttributeError:
            print("No trajectory file (.dcd) was loaded.\n Please load one before using this method.\n")
            return

        tempData        = self.dcdData
        tempdcdFreq     = self.dcdFreq
        tempnbrFrames   = self.nbrFrames
        tempnbrSteps    = self.nbrSteps

        self.importDCDFile(dcdFile)

        #_Append the new data at the end, along the frame axis ('y' axis)
        self.dcdData    = np.append(tempData, self.dcdData, axis=1)
        self.dcdFreq    = np.append(tempdcdFreq, self.dcdFreq)
        self.nbrFrames  += tempData.nbrFrames
        self.nbrSteps   += tempData.nbrSteps


