import numpy as np
import re

from collections import namedtuple


class LOGReader:

    def __init__(self):

        self.logData = None


    def importLOGFile(self, logFile):
        """ Imports a new file and store the result in self.logData.
            If something already exists in self.logData, it will be deleted. """

        #_Open the file and get the lines
        try:
            with open(logFile, 'r', encoding='utf-16') as fileToRead:
                raw_data = fileToRead.read().splitlines()

        except UnicodeError:
            with open(logFile, 'r') as fileToRead:
                raw_data = fileToRead.read().splitlines()

        except:    
            print("Error while reading the file.\nPlease check the file path given in argument.")
            return 

        #_Store each lines corresponding to energies output to a list
        entries = []
        for line in raw_data:
            if re.search('^ENERGY:', line):
                entries.append(line.split()[1:])
            elif re.search('^Info: TIMESTEP', line):
                self.timestep = float(line.split()[2])
            elif re.search('^ETITLE:', line):
                self.etitle = line.split()[1:]


        #_Create a namedtuple so that data are stored in a secured way and can be retrieved using keywords
        dataTuple = namedtuple('dataTuple', " ".join(self.etitle)) 

        #_This dictionary is meant to be used to easily retrieve data series and column labels
        self.keywordsDict = {}
        for i, title in enumerate(self.etitle):
            self.keywordsDict[title] = i   

        #_Convert the python list to numpy array for easier manipulation
        entries = np.array(entries).astype(float)

        #_Store the data in the namedtuple according to their column/keyword
        self.logData = dataTuple(*[ entries[:,col] for col, val in enumerate(entries[0]) ]) 



    def appendLOG(self, logFile):
        """ Method to append output data to the already loaded ones.
            Timestep number is simply continued by adding the last timestep from initial logData if
            the first one is set to zero. """

        try:
            self.logData #_Checking if a log file has been loaded already, print an error message if not.
        except AttributeError:
            print("No output file (.log or .out) was loaded.\n Please load one before using this method.\n")
            return

        tempData = self.logData
    
        self.importLOGData(logFile)

        #_Adding last timestep number and applying correction for possible different timesteps
        if tempData.TS[0] == 0:
            tempData = tempData._replace(TS = self.logData.TS[-1] 
                                                                 + tempData.TS 
                                                                 * (self.timestep 
                                                                 / tempData.timestep) )

        for i, etitle in enumerate(self.logData.etitle):
            self.logData.dataSet = self.logData.dataSet._replace( **{etitle: 
                                                    np.append(self.logData.dataSet[i], tempData.dataSet[i])} )
 
