"""

Classes
^^^^^^^

"""

import sys

import re

from collections import namedtuple

import numpy as np


class HydroproReader:
    """ This class can be used to extract usefule information from a hydropro result file that
        can be used with ScatDiffusion module from NamdAnalyzer. 

        :arg resFile: result file as obtained from HydroPro software

    """

    def __init__(self, resFile=None):

        self.paramsTuple = namedtuple('params', 'temp solViscosity molWeight specVol solDensity dt0 rg vol dr0'
                                                                                +' sedCoeff rh')


        self.params = None

        if resFile:
            self.readFile(resFile)



    def readFile(self, resFile):
        """ Extracts different parameters from hydropro output file using regular expression. 

            Result is stored in *params* namedtuple attribute with the following keys:
                - temp          - temperature used
                - solViscosity  - solvent viscosity at given temperature
                - molWeight     - molecular weight of the molecule
                - specVol       - specific volume
                - solDensity    - solvent density at given temperature
                - dt0           - translational diffusion coefficient
                - rg            - gyration radius
                - dr0           - rotational diffusion coefficient
                - sedCoeff      - sedimentation coefficient
                - rh            - hydrodynamic radius

        """

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


