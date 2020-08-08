"""

Classes
^^^^^^^

"""

import numpy as np

from struct import *


class VELReader:
    """ This class is used to read and extract data from a .vel file. """

    def __init__(self):

        self.velData    = None
        self.nbrAtoms   = None


    def importVELFile(self, velFile):
        """ Imports a new file and store the result in self.velData.
            If something already exists in *velData* attribute,
            it will be deleted.

        """

        self.velFile = velFile

        with open(velFile, 'rb') as f:
            data = f.read()

        self.nbrAtoms = unpack('i', data[:4])[0]

        # Allocate memory for the data extraction
        self.velData = np.zeros((self.nbrAtoms, 3))
        # Read and convert the data to 64-bit float by group of
        # 3 corresponding to (x, y, z) velocities
        # for each atom. The resulting array contains
        # (x, y, z) velocities along axis 1.
        for i in range(self.nbrAtoms):
            self.velData[i] = unpack('ddd', data[24 * i + 4:24 * i + 28])

        data = None
