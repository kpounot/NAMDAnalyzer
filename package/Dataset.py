import os, sys
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from .dataParsers.dcdParser import NAMDDCD
from .dataParsers.logParser import NAMDLOG
from .dataParsers.pdbParser import NAMDPDB
from .dataParsers.psfParser import NAMDPSF
from .dataParsers.velParser import NAMDVEL
from .helpersFunctions.ConstantsAndModels import *

class Dataset(NAMDDCD, NAMDVEL, NAMDLOG, NAMDPDB, NAMDPSF):
    """ This class act as the main controller, allowing the user to import different data types,
        namely trajectories, NAMD log file or velocities. Each datafile is used to initialize a 
        corresponding class, from which the different methods can be called. 
        
        A selection dict (selList) is used to store the user custom atom selections. Methods are available 
        to add or remove selections (newSelection and delSelection). Both of them needs a psf file
        to be loaded so that it can call the getSelection method in self.psfData instance. """


    def __init__(self, fileList):

        NAMDDCD.__init__(self)
        NAMDVEL.__init__(self)
        NAMDLOG.__init__(self)
        NAMDPDB.__init__(self)
        NAMDPSF.__init__(self)

        if isinstance(fileList, str): #_Single call to importFile of fileList is a string
            self.importFile(fileList)
        elif isinstance(fileList, list): #_If fileList is an actual list, call importFile for each entry
            for f in fileList:
                self.importFile(f)


    def importFile(self, dataFile, fileType=None):
        """ Method used to import one file.
            The method automatically stores the corresponding class in NAMDAnalyzer variables like
            self.logData. If something already exists, it will be overridden.

            Input:  a single data file (*.log, *.dcd,...)
                    fileType -> data file type, can be 'log or 'out' for standard NAMD log output, 'dcd',
                                'vel', 'psf' or 'pdb'. If None, the file type will be guessed from extension."""

        print("Trying to import file: " + dataFile)
        try: #_Trying to open the file. Raise an exception if not found for guessing.
            if fileType=="out" or fileType=="log" or re.search('.log|.out', dataFile):
                self.importLOGFile(dataFile)

            elif fileType=="dcd" or re.search('.dcd', dataFile):
                self.importDCDFile(dataFile)

            elif fileType=="pdb" or re.search('.pdb', dataFile):
                self.importPDBFile(dataFile)

            elif fileType=="psf" or re.search('.psf', dataFile):
                self.importPSFFile(dataFile)

            elif fileType=="vel" or re.search('.vel', dataFile):
                self.importVELFile(dataFile)

            else:
                raise Exception("File extension not recognized.")

        except Exception as inst:
            print(inst)
            return

        print("Done\n")

