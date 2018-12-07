import os, sys
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors



class Dataset(NAMDPSF):
    """ This class inherits from a psf file class and is a general one that will be in inherited by
        subsequent file types classes like NAMDDCD, NAMDVEL,... """        

    def __init__(self, fileList):

        psfFile = self.getPSF(fileList) #_Check for presence of a .psf file

        if psfFile: #_If psfFile was found, initialize NAMDPSF
            NAMDPSF.__init__(self, psfFile)






