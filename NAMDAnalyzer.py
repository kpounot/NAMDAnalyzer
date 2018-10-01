import os, sys
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

import dataParsers.logParser as logParser
import dataParsers.dcdParser as dcdParser
import dataParsers.pdbParser as pdbParser
import dataParsers.psfParser as psfParser
import dataParsers.velParser as velParser

import IPython


class NAMDAnalyzer:
    """ This class act as the main controller, allowing the user to import different data types,
        namely trajectories, NAMD log file or velocities. Each datafile is used to initialize a 
        corresponding class, from which the different methods can be called. """


    def __init__(self, fileList):

        if isinstance(fileList, str): #_Single call to importFile of fileList is a string
            self._importFile(fileList)
        elif isinstance(fileList, list): #_If fileList is an actual list, call importFile for each entry
            for f in fileList:
                self._importFile(f)

        #_Defines some constants and formulas
        self.kB_kcal = 0.00198720

        self.fMaxBoltzDist = lambda x, T: ( 1 / np.sqrt(np.pi * (T * self.kB_kcal)**3) 
                                                    * np.sqrt(x) * np.exp(-x/(self.kB_kcal * T)) ) 

        self.fgaussianModel = lambda x, a, b, c: a * np.exp(-(x-b)**2/c**2)
                

    def _importFile(self, dataFile):
        """ Method used to import one file.
            The method automatically stores the corresponding class in NAMDAnalyzer variables like
            self.logData. If something already exists, it will be overridden.

            Input: a single data file (*.log, *.dcd,...) """

        print("Trying to import file: " + dataFile)
        try: #_Trying to guess the file type. Raise an exception if not found.
            if re.search('.log', dataFile):
                self.logData = logParser.NAMDLog(dataFile, parent=self)
            elif re.search('.dcd', dataFile):
                self.dcdData = dcdParser.NAMDDCD(dataFile, parent=self)
            elif re.search('.pdb', dataFile):
                self.pdbData = pdbParser.NAMDPDB(dataFile, parent=self)
            elif re.search('.psf', dataFile):
                self.psfData = psfParser.NAMDPSF(dataFile, parent=self)
            elif re.search('.vel', dataFile):
                self.velData = velParser.NAMDVel(dataFile, parent=self)
            else:
                raise Exception("File extension not recognized.")

            print("File successfully loaded\n")

        except Exception as inst:
            print(type(inst))
            print(inst.args)



if __name__ == '__main__':

    fileList = sys.argv[1:]

    data = NAMDAnalyzer(fileList)

    shell = IPython.terminal.embed.InteractiveShellEmbed()
    shell.enable_gui()
    shell.enable_matplotlib()

    shell()

        

