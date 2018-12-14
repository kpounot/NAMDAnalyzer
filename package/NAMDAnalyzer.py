import os, sys
import numpy as np
import IPython

import re


#from .dataConverters.backscatteringDataConvert import BackScatData
from .test.interFuncTest import BackScatData
from .dataParsers.logParser import NAMDLOG
from .dataParsers.pdbParser import NAMDPDB
from .dataParsers.velParser import NAMDVEL
from .dataParsers.psfParser import NAMDPSF

class NAMDAnalyzer(NAMDPSF):
    """ Main class for NAMDAnalyzer. It manages the different data types (psf, dcd, vel,...) """

    def __init__(self, fileList):

        if isinstance(fileList, str):
            self.fileList = [fileList]
        else:
            self.fileList = fileList

        self.psfFile = self.getPSF(self.fileList) #_Check for .psf file presence

        NAMDPSF.__init__(self, self.psfFile) #_Initialize NAMDPSF to access selection methods directly from here

        self.logData = NAMDLOG(self)
        self.dcdData = BackScatData(self)
        self.velData = NAMDVEL(self)
        self.pdbData = NAMDPDB(self)

        for f in self.fileList:
            self.importFile(f)



    def getPSF(self, fileList):
        """ This method checks for a psf file in the file list given as __init__ arguments.
            Returns the .psf file path if found, returns None otherwise. """

        try:
            for idx, dataFile in enumerate(fileList):
                if re.search('.psf', dataFile):
                    self.fileList.pop(idx)
                    return dataFile

            raise Exception("No .psf file found. Please load one to initialize NAMDAnalyzer.")
        
        except Exception as inst:
            print(inst)
            return



    def importFile(self, dataFile, fileType=None):
        """ Method used to import one file.
            The method automatically stores the corresponding class in NAMDAnalyzer variables like
            self.logData. If something already exists, it will be overridden.

            Input:  a single data file (*.log, *.dcd,...)
                    fileType -> data file type, can be 'log or 'out' for standard NAMD log output, 'dcd',
                                'vel' or 'pdb'. If None, the file type will be guessed from extension."""

        print("Trying to import file: " + dataFile)
        if fileType=="out" or fileType=="log" or re.search('.log|.out', dataFile):
            self.logData.importLOGFile(dataFile)

        elif fileType=="dcd" or re.search('.dcd', dataFile):
            self.dcdData.importDCDFile(dataFile)

        elif fileType=="pdb" or re.search('.pdb', dataFile):
            self.pdbData.importPDBFile(dataFile)

        elif fileType=="vel" or re.search('.vel', dataFile):
            self.velData.importVELFile(dataFile)

        elif fileType=="psf" or re.search('.psf', dataFile):
            print("Nothing to do, .psf file should be already loaded.")
            return

        elif dataFile == [] or dataFile == None:
            return

        else:
            raise Exception("File extension not recognized.")


        print("Done\n")



if __name__ == '__main__':

    fileList = sys.argv[1:]

    data = NAMDAnalyzer(fileList)

        
