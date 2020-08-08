"""

Classes
^^^^^^^

"""

import numpy as np
import re

from collections import namedtuple


class PSFReader:

    def __init__(self):

        self.psfData = None


    def importPSFFile(self, psfFile):
        """ Imports a new file and store the result in self.psfData.
            If something already exists in self.psfData, it will be deleted.

        """

        self.psfFile = psfFile

        with open(psfFile, 'r') as f:
            data = f.read().splitlines()

        dataTuple = namedtuple('dataTuple', 'atoms bonds angles dihedrals '
                                            'impropers donors acceptors '
                                            'nonBonded xterms')

        # Initialize the temporary lists
        atomList        = []
        bondList        = []
        thetaList       = []
        phiList         = []
        imprpList       = []
        donorsList      = []
        acceptorsList   = []
        nonBondedList   = []
        xTermsList      = []

        pattern = re.compile('\s+[0-9]+\s![A-Z]+')  # Entry category id
        for i, line in enumerate(data):
            if pattern.match(line):
                # Get the number of lines to read for the category
                # Each category data is stored in a corresponding temporary
                # array, which will be then copied and protected from
                # modifications in the namedtuple.
                nbrEntries = int(line.split()[0])
                if re.search('ATOM', line):
                    atomList = np.array([entry.split() for entry
                                         in data[i + 1:i + 1 + nbrEntries]])
                elif re.search('BOND', line):
                    bondList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + int(nbrEntries / 4)]])
                    bondList = bondList.astype(int)
                elif re.search('THETA', line):
                    thetaList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + int(nbrEntries / 3)]])
                    thetaList = thetaList.astype(int)
                elif re.search('PHI', line):
                    phiList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + int(nbrEntries / 2)]])
                    phiList = phiList.astype(int)
                elif re.search('IMPHI', line):
                    imprpList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + int(nbrEntries / 2)]])
                    imprpList = imprpList.astype(int)
                elif re.search('DON', line):
                    donorsList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + nbrEntries]])
                elif re.search('ACC', line):
                    acceptorsList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + nbrEntries]])
                elif re.search('NNB', line):
                    nonBondedList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + nbrEntries]])
                elif re.search('CRTERM', line):
                    xTermsList = np.array(
                        [entry.split() for entry
                         in data[i + 1:i + 1 + int(nbrEntries / 2)]])
                    xTermsList = xTermsList.astype(int)

        self.psfData = dataTuple(
            atomList, bondList, thetaList, phiList, imprpList,
            donorsList, acceptorsList, nonBondedList, xTermsList)
