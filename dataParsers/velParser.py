import os, sys
import numpy as np
import re

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from struct import *

class NAMDVel:
    """ This class contains methods for velocity file analysis. """

    def __init__(self, velFile, parent=None):

        self.parent = parent
        
        with open(velFile, 'rb') as f:
            data = f.read()

        self.nbrAtoms = unpack('i', data[:4])[0]

        #_Allocate memmory for the data extraction
        self.dataSet = np.zeros((self.nbrAtoms, 3))
        #_Read and convert the data to 64-bit float by group of 3 corresponding to (x, y, z) velocities
        #_for each atom. The resulting array contains (x, y, z) velocities along axis 1.
        for i in range(self.nbrAtoms):
            self.dataSet[i] = unpack('ddd', data[24*i+4:24*i+28])

        
        
    #---------------------------------------------
    #_Data accession methods
    #---------------------------------------------

    def getKineticEnergyDistribution(self, binSize=0.2, begin=0, end=None):
        """ This method can be used to compute the kinetic energy distribution without plotting it.
            This method requires a .psf to be loaded to have access to atom masses

            Input:  binSize -> the size of the bin
                    begin   -> start index for the data series
                    end     -> end index for the data series

            Output: numpy 2D array containing the bin mean value (first column)
                    and the corresponding density (second column) """

        try: #_Try to get the masses list for each selected atoms
            massList = self.parent.psfData.getAtomsMasses(begin=begin, end=end) 
        except AttributeError:
            print("No psf data can be found in the NAMDAnalyzer object.\n Please load a .psf file.\n")
            return

        #_Computes the kinetic energy for each atom
        data = 0.5 * massList * np.sum(self.dataSet**2, axis=1)[begin:end] 
        data = np.sort(data)

        #_Defining the locations for binning
        xBins = np.arange(data.min(), data.max() + 1, binSize)

        density = np.zeros(xBins.shape[0])

        #_Start accumulation in density vector, first we create an iterator on data array
        it = np.nditer(data)
        #_Then we loop on the bins
        for i, val in enumerate(xBins):
            #_While we're not at the end of the iterator, we keep iterating
            while it.iternext():
                #_Comparison of iterator value with high bin limit, if lower, we increase density index by 1
                if it.value < val:
                    density[i] += 1
                else:
                    break #_Stops the iteration if the value is larger and continue to the next bin

        #_Adding one to each entry of the density to compensate for the first missing increment, which
        #_is due to the while loop.
        density += 1

        #_Finally we normalize the density vector
        density /= np.sum(density)

        return np.column_stack((xBins, density))

 

    #---------------------------------------------
    #_Plotting methods
    #---------------------------------------------

    def plotKineticEnergyDistribution(self, binSize=0.1, begin=0, end=None, fit=False, model=None, p0=None):
        """ This method calls pylab's hist method is used to plot the distribution.
            !!! This method requires a .psf to be loaded to have access to atom masses !!!

            Input:  binSize -> the size of the bin. Determine the width of each rectangle of the histogram 
                    begin   -> start index for data
                    end     -> end index for data
                    fit     -> if set to True, use the given model in scipy's curve_fit method and plot it
                    model   -> model to be used for the fit """

        try: #_Try to get the masses list for each selected atoms
            massList = self.parent.psfData.getAtomsMasses(begin=begin, end=end) 
        except AttributeError:
            print("No psf data can be found in the NAMDAnalyzer object.\n Please load a .psf file.\n")
            return

        #_Computes the kinetic energy for each atom
        data = 0.5 * massList * np.sum(self.dataSet**2, axis=1)[begin:end] 
 
        #_Defining the locations for binning
        xBins = np.arange(data.min(), data.max() + 1, binSize) 
    
        fig, ax = plt.subplots()

        ax.hist(data, bins=xBins, density=True, edgecolor='black')
        ax.set_xlabel(r'$Kinetic \ energy \ (kcal/mol)$')
        ax.set_ylabel('Density')

        if fit:
            if model is None:
                print("Fit model error: fit was set to 'True' but no model was given.\n")
                return

            #_Extract data directly from the plot (using the first entry in case of subplots), and call
            #_scipy's curve_fit procedure with the given model. Then plot the fitted model on top.
            plotData = np.array([ (patch.get_x(), patch.get_height()) for patch in ax.patches ])
            params = curve_fit(model, plotData[:,0], plotData[:,1], p0=p0)
            ax.plot(plotData[:,0], model(plotData[:,0], *params[0]), color='orange', label='Fit params:\n' + 
                                                            "\n".join(params[0].astype(str)))
            plt.legend(framealpha=0.5)

            plt.tight_layout()
            return plt.show(block=False)

        else:
            plt.tight_layout()
            return plt.show(block=False)

