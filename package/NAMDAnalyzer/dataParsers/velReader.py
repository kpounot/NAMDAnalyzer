import numpy as np

from struct import *

class VELReader:

    def __init__(self):

        self.velData    = None
        self.nbrAtoms   = None


    def importVELFile(self, velFile):
        """ Imports a new file and store the result in self.velData.
            If something already exists in self.velData, it will be deleted. """

        self.velFile = velFile

        with open(velFile, 'rb') as f:
            data = f.read()

        self.nbrAtoms = unpack('i', data[:4])[0]

        #_Allocate memory for the data extraction
        self.velData = np.zeros((self.nbrAtoms, 3))
        #_Read and convert the data to 64-bit float by group of 3 corresponding to (x, y, z) velocities
        #_for each atom. The resulting array contains (x, y, z) velocities along axis 1.
        for i in range(self.nbrAtoms):
            self.velData[i] = unpack('ddd', data[24*i+4:24*i+28])

        data = None

