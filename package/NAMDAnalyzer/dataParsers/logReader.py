"""

Classes
^^^^^^^

"""

import numpy as np
import re

from collections import namedtuple


class LOGReader:
    """ Class used to read and extract data from a .log/.out file which
        is simply the standard NAMD output.

    """

    def __init__(self):

        self.logData = None


    def importLOGFile(self, logFile):
        """ Imports a new file and store the result in self.logData.
            If something already exists in self.logData, it will be deleted.

        """

        self.logFile = logFile

        # Open the file and get the lines
        try:
            with open(logFile, 'r', encoding='utf-16') as fileToRead:
                raw_data = fileToRead.read().splitlines()

        except UnicodeError:
            with open(logFile, 'r') as fileToRead:
                raw_data = fileToRead.read().splitlines()

        except Exception as e:
            print("Error while reading the file.\n"
                  "Please check the file path given in argument.")
            print(e)
            return

        # Store each lines corresponding to energies output to a list
        entries = []
        for line in raw_data:
            if re.search('^ENERGY:', line):
                entries.append(line.split()[1:])
            elif re.search('^Info: TIMESTEP', line):
                self.timestep = float(line.split()[2]) * 1e-15
            elif re.search('^ETITLE:', line):
                self.etitle = line.split()[1:]


        # Create a namedtuple so that data are stored in a secured
        # way and can be retrieved using keywords
        dataTuple = namedtuple('dataTuple', " ".join(self.etitle))

        # This dictionary is meant to be used to easily retrieve data
        # series and column labels
        self.keywordsDict = {}
        for i, title in enumerate(self.etitle):
            self.keywordsDict[title] = i

        # Convert the python list to numpy array for easier manipulation
        entries = np.array(entries).astype(float)

        # Store the data in the namedtuple according to their column/keyword
        self.logData = dataTuple(
            *[entries[:, col] for col, val in enumerate(entries[0])])



    def appendLOG(self, logFile):
        """ Method to append output data to the already loaded ones.
            Timestep number is simply continued by adding the last
            timestep from initial logData if the first one is set to zero.

        """

        try:
            self.logData  # Checking if a log file has been loaded already.
        except AttributeError:
            print("No output file (.log or .out) was loaded.\n"
                  "Please load one before with importLOGFile "
                  "using this method.\n")
            return

        tempData  = self.logData
        old_tstep = self.timestep

        self.importLOGFile(logFile)

        # Adding last timestep number and applying correction
        # for possible different timesteps
        if tempData.TS[0] == 0:
            tempData = tempData._replace(TS = self.logData.TS[-1]
                                         + tempData.TS
                                         * (self.timestep
                                         / old_tstep))

        for i, etitle in enumerate(self.etitle):
            self.logData = self.logData._replace(
                **{etitle: np.append(self.logData[i],
                                     tempData[i])})
