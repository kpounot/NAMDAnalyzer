import os, sys
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy.io import FortranFile

from collections import namedtuple


class NAMDDCD:
    """ This class contains methods for trajectory file analysis. """

    def __init__(self, dcdfile, parent=None):

        self.parent = parent

        try:
            dcdFile = FortranFile(dcdfile, 'r')
        except:
            print("Error while reading the file.\n" + "Please check file path/extension.")
            return

        #_Get some simulation parameters (frames, steps and dcd frequency)
        header = dcdFile.read_record('i4')
        self.nbrFrames  = header[1]
        self.dcdFreq    = header[3]
        self.nbrSteps   = header[4]

        #_Skip the next record
        dcdFile.read_record(dtype='i4')

        #_Get the number of atoms
        self.nbrAtoms = dcdFile.read_record(dtype='i4')[0]

        #_Read data for all frames and store coordinated in self.dataSet 
        self.dataSet = np.zeros((self.nbrAtoms, 3*self.nbrFrames))
        for i in range(3 * self.nbrFrames):
            self.dataSet[:,i] = dcdFile.read_record('f')

        #_The dataset is reshaped so that we have atom index in axis 0, frame number in axis 1, 
        #_and (x, y, z) coordinates in axis 3
        self.dataSet = self.dataSet.reshape(self.dataSet.shape[0], self.nbrFrames, 3)

        
    #_____________________________________________
    #_Data accession methods
    #_____________________________________________

    def getRMSDperAtom(self, begin=0, end=None, mergeXYZ=True):
        """ Computes the standard deviation along the axis 1 of dataSet.

            If mergeXYZ is True, then it computes the distance to the origin first. """

        if mergeXYZ:
            rmsd = np.sqrt(np.sum(self.dataSet[begin:end]**2, axis=2))
            rmsd = np.std(rmsd, axis=1)

        else:
            rmsd = np.std(self.dataSet[begin:end], axis=1)

        return rmsd

    #_____________________________________________
    #_Plotting methods
    #_____________________________________________

    def plotRMSDperAtom(self, begin=0, end=None, mergeXYZ=True):
        """ Plot the standard deviation along the axis 1 of dataSet.
            This makes use of the 'getRMSDperAtom method.

            If mergeXYZ is True, then it computes the distance to the origin first. """

        rmsd = self.getRMSDperAtom(begin=begin, end=end, mergeXYZ=mergeXYZ)
        xRange = np.arange(rmsd.shape[0])

        if mergeXYZ:
            plt.plot(xRange, rmsd)
            plt.ylabel(r'$RMSD \ (\AA)$')

        else:
            #_In case of three columns for (x, y, z) coordinates, generate three plot for each.
            fig, ax = plt.subplots(3, 1, sharex=True)

            ax[0].plot(xRange, rmsd[:,0])
            ax[0].set_ylabel(r'$RMSD \ along \ X \ (\AA)$')

            ax[1].plot(xRange, rmsd[:,1])
            ax[1].set_ylabel(r'$RMSD \ along \ Y \ (\AA)$')

            ax[2].plot(xRange, rmsd[:,2])
            ax[2].set_ylabel(r'$RMSD \ along \ Z \ (\AA)$')

        plt.xlabel(r'$Atom \ index$')

        plt.tight_layout()
        plt.show(block=False)
