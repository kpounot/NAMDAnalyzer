"""

Classes
^^^^^^^

"""

import os, sys
import numpy as np
import re

from NAMDAnalyzer.dataParsers.pdbReader import PDBReader
from NAMDAnalyzer.dataParsers.psfParser import NAMDPSF

class NAMDPDB(PDBReader):
    """ This class takes a PDB file as input.
        The different types of entries (ATOM, HETATOM,...) are stored in separate lists of lists.
        In case of an 'TER' or a water, the new chain's atoms list is simply append to the main list. 

    """

    def __init__(self, pdbFile=None):

        PDBReader.__init__(self)

        if pdbFile:
            self.importPDBFile(pdbFile)

           
    #---------------------------------------------
    #_Data accession methods
    #---------------------------------------------

    def getCoor(self, chainIdx=0):
        """ Extract coordinates from pdb data.
            
            :arg chainIdx: index of the wanted chain in self.atomList (optional, default 0) 

        """

        coor = np.zeros( (self.atomList[chainIdx].shape[0], 3) )

        for i in range(self.atomList[chainIdx].shape[0]):
            coor[i] = np.array( self.atomList[chainIdx][i].split()[6:9] )


        return coor



    #---------------------------------------------
    #_Plotting methods
    #---------------------------------------------


