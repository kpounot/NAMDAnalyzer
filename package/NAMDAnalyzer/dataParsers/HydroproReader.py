import sys

import re

from collections import namedtuple

import numpy as np


class HydroproReader:

    def __init__(self, resFile=None):
        """ This class can be used to extract usefule information from a hydropro result file that
            can be used with ScatDiffusion module from NamdAnalyzer. """

        self.paramsTuple = namedtuple('params', 'temp solViscosity molWeight specVol solDensity dt0 rg vol dr0'
                                                                                +' sedCoeff rh')


        self.params = None

        if resFile:
            self.readFile(resFile)



    def readFile(self, resFile):
        """ Extracts different parameters from hydropro output file using regular expression. """

        #_Temporary list to store parameters
        paramsList = []

        with open(resFile, 'r') as f:
            res = f.readlines()


        for line in res:
            if re.search('\s*Temperature:', line):
                paramsList.append( 273.2 + float(line.split()[1]) )
            if re.search('\s*Solvent viscosity:', line):
                paramsList.append(  float(line.split()[2]) )
            if re.search('\s*Molecular weight:', line):
                paramsList.append( float(line.split()[2]) )
            if re.search('\s*Solute partial specific volume:', line):
                paramsList.append( float(line.split()[4]) )
            if re.search('\s*Solution density:', line):
                paramsList.append( float(line.split()[2]) )
            if re.search('\s*Translational diffusion coefficient:', line):
                paramsList.append( float(line.split()[3]) )
            if re.search('\s*Radius of gyration:', line):
                paramsList.append( float(line.split()[3]) )
            if re.search('\s*Volume:', line):
                paramsList.append( float(line.split()[1]) )
            if re.search('\s*Rotational diffusion coefficient:', line):
                paramsList.append( float(line.split()[3]) )
            if re.search('\s*Sedimentation coefficient:', line):
                paramsList.append( float(line.split()[2]) )
            if re.search('\s*Translational:', line):
                paramsList.append( float(line.split()[1]) )


        self.params = self.paramsTuple(*paramsList)


