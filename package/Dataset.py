import os, sys
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from .dataParsers import *

class Dataset:
    """ This class act as the main controller, allowing the user to import different data types,
        namely trajectories, NAMD log file or velocities. Each datafile is used to initialize a 
        corresponding class, from which the different methods can be called. 
        
        A selection dict (selList) is used to store the user custom atom selections. Methods are available 
        to add or remove selections (newSelection and delSelection). Both of them needs a psf file
        to be loaded so that it can call the getSelection method in self.psfData instance. """


    def __init__(self, fileList):

        if isinstance(fileList, str): #_Single call to importFile of fileList is a string
            self.importFile(fileList)
        elif isinstance(fileList, list): #_If fileList is an actual list, call importFile for each entry
            for f in fileList:
                self.importFile(f)

        self.selList = dict()

        #_Defines some constants and formulas
        self.kB_kcal = 0.00198657
        self.fMaxBoltzDist = lambda x, T: ( 2 / np.sqrt(np.pi * (T * self.kB_kcal)**3) 
                                                    * np.sqrt(x) * np.exp(-x/(self.kB_kcal * T)) ) 
        self.fgaussianModel = lambda x, a, b, c: a / (np.sqrt(2*np.pi) * c) * np.exp(-(x-b)**2 / (2*c**2))

        #_Ideal resolution for SHPERES instrument, FWHM of 0.65e-6 eV (single gaussian)
        self.resFuncSPHERES = lambda x: np.exp(-x**2/(2*0.276e-6**2))
                

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
                self.logData = logParser.NAMDLog(dataFile, parent=self)
            elif fileType=="dcd" or re.search('.dcd', dataFile):
                self.dcdData = dcdParser.NAMDDCD(dataFile, parent=self)
            elif fileType=="pdb" or re.search('.pdb', dataFile):
                self.pdbData = pdbParser.NAMDPDB(dataFile, parent=self)
            elif fileType=="psf" or re.search('.psf', dataFile):
                self.psfData = psfParser.NAMDPSF(dataFile, parent=self)
            elif fileType=="vel" or re.search('.vel', dataFile):
                self.velData = velParser.NAMDVel(dataFile, parent=self)
            else:
                raise Exception("File extension not recognized.")

        except Exception as inst:
            print(inst)
            return

        print("Done\n")


    def appendLog(self, logFile):
        """ Method to append output data to the already loaded ones.
            Timestep number is simply continued by adding the last timestep from initial logData if
            the first one is set to zero. """

        try:
            self.logData #_Checking if a dcd file has been loaded already, print an error message if not.
        except AttributeError:
            print("No output file (.log or .out) was loaded.\n Please load one before using this method.\n")
            return

        tempData = logParser.NAMDLog(logFile, parent=self)
    
        #_Adding last timestep number and applying correction for possible different timesteps
        if tempData.dataSet.TS[0] == 0:
            tempData.dataSet = tempData.dataSet._replace(TS = self.logData.dataSet.TS[-1] 
                                                                 + tempData.dataSet.TS 
                                                                 * (tempData.timestep 
                                                                    / self.logData.timestep) )

        for i, etitle in enumerate(self.logData.etitle):
            self.logData.dataSet = self.logData.dataSet._replace( **{etitle: 
                                                    np.append(self.logData.dataSet[i], tempData.dataSet[i])} )
            
    def appendDCD(self, dcdFile):
        """ Method to append trajectory data to the existing loaded data.
            The .dcd file is opened using dcdParser class, then data are append to the existing
            dcdData instance. 

            Input:  a single .dcd trajectory file """
        
        try:
            self.dcdData #_Checking if a dcd file has been loaded already, print an error message if not.
        except AttributeError:
            print("No trajectory file (.dcd) was loaded.\n Please load one before using this method.\n")
            return

        tempData = dcdParser.NAMDDCD(dcdFile, parent=self)

        #_Append the new data at the end, along the frame axis ('y' axis)
        self.dcdData.dataSet    = np.append(self.dcdData.dataSet, tempData.dataSet, axis=1)
        self.dcdData.dcdFreq    = np.append(self.dcdData.dcdFreq, tempData.dcdFreq)
        self.dcdData.nbrFrames  += tempData.nbrFrames
        self.dcdData.nbrSteps   += tempData.nbrSteps


    def appendVEL(self, velFile):
        """ Method to append velocities data to the existing loaded data.
            The .vel file is opened using velParser class, then data are append to the existing
            velData instance. 

            Input:  a single .vel trajectory file """
        
        try:
            self.velData #_Checking if a dcd file has been loaded already, print an error message if not.
        except AttributeError:
            print("No velocities file (.vel) was loaded.\n Please load one before using this method.\n")
            return

        tempData = velParser.NAMDVel(velFile, parent=self)

        #_Append the new data at the end, along the frame axis ('y' axis)
        self.velData.dataSet    = np.append(self.velData.dataSet, tempData.dataSet, axis=0)




    def newSelection(self, selName, selText="all", segName=None, NOTsegName=None, resNbr=None, NOTresNbr=None,
                        resName=None, NOTresName=None, atom=None, NOTatom=None, index=None):
        """ Calls the self.psfData.getSelection method and store the list of selected indices 
            in the self.selList attribute. """

        self.selList[selName] = self.psfData.getSelection(selText, segName, NOTsegName, resNbr, NOTresNbr, 
                        resName, NOTresName, atom, NOTatom, index) 


    def delSelection(self, selName):
        """ Remove the selection from self.selList at the given index. """

        self.selList.pop(selName)


        
