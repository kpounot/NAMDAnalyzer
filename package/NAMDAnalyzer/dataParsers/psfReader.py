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

        # neutron incoherent scattering cross-section 
        # for most common isotopes in biophysics
        self.dict_nScatLength_inc = {'H': 25.274,
                                     '2H': 4.04,
                                     'C': 0,
                                     '13C': -0.52,
                                     'N': 2.0,
                                     '15N': -0.02,
                                     'O': 0.0,
                                     'P': 0.2,
                                     'S': 0.0}
                 

        # neutron coherent scattering cross-section 
        # for most common isotopes in biophysics
        self.dict_nScatLength_coh = {'H': -3.7406,
                                     '2H': 6.671,
                                     'C': 6.6511,
                                     '13C': 6.19,
                                     'N': 9.37,
                                     '15N': 6.44,
                                     'O': 5.803,
                                     'P': 5.13,
                                     'S': 2.804}
                          

    def importPSFFile(self, psfFile):
        """ Imports a new file and store the result in self.psfData.
            If something already exists in self.psfData, it will be deleted.

        """

        self.psfFile = psfFile

        with open(psfFile, 'r') as f:
            data = f.read().splitlines()

        dataTuple = namedtuple('dataTuple', 'atoms bonds angles dihedrals '
                                            'impropers donors acceptors '
                                            'nonBonded xterms '
                                            'nScatLength_inc '
                                            'nScatLength_coh')

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


        nScatLength_inc, nScatLength_coh = self._setScatLength(atomList)
        


        self.psfData = dataTuple(
            atomList, bondList, thetaList, phiList, imprpList,
            donorsList, acceptorsList, nonBondedList, xTermsList,
            nScatLength_inc, nScatLength_coh)



    def _setScatLength(self, atomList):
        """ This method aims at setting two arrays which
            contain the scattering length for each atom in the psf file.

            The value is the one for the most common isotope by default,
            but it can be changed later using a selection (see
            :class:`selText`)

        """

        nScatLength_inc = np.zeros(atomList.shape[0])
        nScatLength_coh = np.zeros(atomList.shape[0])

        for idx, atom in enumerate(atomList):
            try:
                nScatLength_inc[idx] = self.dict_nScatLength_inc[atom[4][0]]
                nScatLength_coh[idx] = self.dict_nScatLength_coh[atom[4][0]]
            except KeyError:
                print("Atom %s at index %i could not be assigned a scattering "
                      "length." % (atom, idx))
                


        return nScatLength_inc, nScatLength_coh
