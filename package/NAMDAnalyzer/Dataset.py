import sys

import re

from NAMDAnalyzer.selection.selText import SelText

from NAMDAnalyzer.dataParsers.dcdParser import NAMDDCD
from NAMDAnalyzer.dataParsers.logParser import NAMDLOG
from NAMDAnalyzer.dataParsers.pdbParser import NAMDPDB
from NAMDAnalyzer.dataParsers.velParser import NAMDVEL

import numpy as np


class Dataset(NAMDDCD):
    """ Main class for NAMDAnalyzer.
        It manages the different data types (psf, dcd, vel,...)

        It directly inherits from :class:`.NAMDPSF` and
        :class:`NAMDDCD` so that all methods from these are directly
        accessible in :class:`.Dataset` class.

        Attributes *logData*, *velData*, *dcdData*, *psfData* and
        *pdbData* are available to access dataset loaded rom these files.

        For more information, see:
            * :class:`.NAMDDCD`
            * :class:`.NAMDPSF`
            * :class:`.NAMDLOG`
            * :class:`.NAMDPDB`
            * :class:`.NAMDVEL`

    """

    def __init__(self, *fileList):

        self.fileList = list(fileList)

        # Check for .psf file presence
        self.psfFile = self._getPSF(self.fileList)

        NAMDDCD.__init__(self, self.psfFile)  # Initialize NAMDDCD

        self.logData = NAMDLOG()
        self.velData = NAMDVEL(self)
        self.pdbData = NAMDPDB()

        for f in self.fileList:
            self.importFile(f)

    def _getPSF(self, fileList):
        """ This method checks for a psf file in the file list given
            as __init__ arguments.
            Returns the .psf file path if found, returns None otherwise.

        """

        try:
            for idx, dataFile in enumerate(fileList):
                if re.search('.psf', dataFile):
                    self.fileList.pop(idx)
                    return dataFile

            raise Exception("No .psf file found. Methods related to "
                            "trajectories won't work properly.")

        except Exception as inst:
            print(inst)
            return


    def importFile(self, dataFile, fileType=None):
        """ Method used to import one file.

            The method automatically stores the corresponding class in
            NAMDAnalyzer variables like self.logData. If something already
            exists, it will be overridden.

            :arg dataFile: a single data file (*.log, *.dcd,...)
            :arg fileType: data file type, can be 'log or 'out' for standard
                           NAMD log output, 'dcd', 'vel' or 'pdb'. If None,
                           the file type will be guessed from extension.

        """

        print("Trying to import file: " + dataFile)
        if (fileType == "out" or fileType == "log"
                or re.search('.log|.out', dataFile)):
            self.logData.importLOGFile(dataFile)

        elif fileType == "dcd" or re.search('.dcd', dataFile):
            self.importDCDFile(dataFile)

        elif fileType == "pdb" or re.search('.pdb', dataFile):
            self.pdbData.importPDBFile(dataFile)

        elif fileType == "vel" or re.search('.vel', dataFile):
            self.velData.importVELFile(dataFile)

        elif fileType == "psf" or re.search('.psf', dataFile):
            self.psfFile = dataFile
            self.importPSFFile(dataFile)

        elif dataFile == [] or dataFile is None:
            return

        else:
            raise Exception("File extension not recognized.")

        print("Done\n")


    def selection(self, selT='all'):
        """ Uses the :class:`Selection` class to select atoms
            with a simple string command.

            Default frame is the last one.

            :arg selText: a selection string (default 'all')

            :returns: a :class:`Selection` class instance


        """

        return SelText(self, selT)


if __name__ == '__main__':

    fileList = sys.argv[1:]

    data = Dataset(*fileList)
