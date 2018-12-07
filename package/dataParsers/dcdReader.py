import numpy as np
import re

from scipy.io import FortranFile
from struct import *


class DCDReader:
    def __init__(self):

        self.dcdData    = None
        self.nbrFrames  = None
        self.timestep   = None
        self.nbrSteps   = None
        self.nbrAtoms   = None
        self.dcdFreq    = None


    def importDCDFile(self, dcdFile):
        """ Imports a new file and store the result in self.dcdData.
            If something already exists in self.dcdData, it will be deleted. """

        self.dcdFile = dcdFile

        self.dcdData = None #_Free memory in case data were already loaded

        dcdFile = FortranFile(dcdFile, 'r')

        #_Get some simulation parameters (frames, steps and dcd frequency)
        header = dcdFile.read_record('i4')
        self.nbrFrames  = header[1]
        dcdFreq         = header[3]
        self.nbrSteps   = header[4]
        self.timestep   = unpack('f', pack('i', header[10]))[0] * 0.04888e-12 

        #_Skip the next record
        dcdFile.read_record(dtype='i4')

        #_Get the number of atoms
        self.nbrAtoms = dcdFile.read_record(dtype='i4')[0]

        #_Allocating memory to make the process much faster
        self.dcdData = np.zeros((self.nbrAtoms, 3*self.nbrFrames))

        #_Check the first entry to determine how data are organized
        firstEntry = dcdFile.read_record('f')

        if firstEntry.shape == self.nbrAtoms:
            self.dcdData[:,0] = firstEntry
            #_Read data for all frames and store coordinated in self.dcdData 
            for i in range(3 * self.nbrFrames - 1):
                self.dcdData[:,i+1] = dcdFile.read_record('f')

        else: #_We need to skip each four lines
            self.dcdData[:,0] = dcdFile.read_record('f')
            self.dcdData[:,1] = dcdFile.read_record('f')
            self.dcdData[:,2] = dcdFile.read_record('f')
            for i in range(self.nbrFrames - 1):
                dcdFile.read_record('f')
                self.dcdData[:,3 + 3*i] = dcdFile.read_record('f')
                self.dcdData[:,4 + 3*i] = dcdFile.read_record('f')
                self.dcdData[:,5 + 3*i] = dcdFile.read_record('f')

        #_The dataset is reshaped so that we have atom index in axis 0, frame number in axis 1, 
        #_and (x, y, z) coordinates in axis 2
        self.dcdData = self.dcdData.reshape(self.dcdData.shape[0], self.nbrFrames, 3)

        #_Converting dcdFreq to an array of size nbrFrames for handling different dcdFreq 
        #_during conversion to time
        self.dcdFreq = np.zeros(self.dcdData.shape[1]) + dcdFreq



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


