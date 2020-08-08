"""

Classes
^^^^^^^

"""

import os
import sys
import numpy as np
import re

from NAMDAnalyzer.dataParsers.pdbReader import PDBReader
from NAMDAnalyzer.dataParsers.psfParser import NAMDPSF


class NAMDPDB(PDBReader):
    """ This class takes a PDB file as input.
        The different types of entries (ATOM, HETATOM,...) are
        stored in separate lists of lists. In case of an 'TER' or a water,
        the new chain's atoms list is simply append to the main list.

    """

    def __init__(self, pdbFile=None):

        PDBReader.__init__(self)

        if pdbFile:
            self.importPDBFile(pdbFile)


# --------------------------------------------
# Data accession methods
# --------------------------------------------

    def getCoor(self, chainIdx=None):
        """ Extract coordinates from pdb data.

            :arg chainIdx: indices of the wanted chain(s)
                           in self.atomList
                           (optional, default None, all chains)

        """

        if chainIdx is None:
            atoms = np.concatenate(self.atomList)
        else:
            atoms = self.atomList[chainIdx]

        coor = np.array([[atoms[i][30:38], atoms[i][38:46], atoms[i][46:54]]
                         for i in range(atoms.shape[0])]).astype('float32')


        return coor
