import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D



class Dynamics:

    def __init__(self, data):
        """ This class defines methods to analyze dynamics of molecules in the system.
            It contains tools to monitor hydrogen bonds, orientational distribution and survival
            probabilities.

            Input: self.data -> a Dataset class instance containing topology and trajectories data """

        self.data = data



#---------------------------------------------
#_Rotational stuff
#---------------------------------------------
    def rotationalRelaxation(self, corrType='continuous'):
        return

    def angularDistribution(self, corrType='continuous'):
        return

#---------------------------------------------
#_Translational stuff
#---------------------------------------------
    def getMSD(self, corrType='continuous'):
        return

    def confinementTime(self, corrType='continuous'):
        return


#---------------------------------------------
#_Plotting methods
#---------------------------------------------

