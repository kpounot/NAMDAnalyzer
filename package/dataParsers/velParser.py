import os, sys
import numpy as np
import re

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from.velReader import VELReader
from .psfParser import NAMDPSF

class NAMDVEL(NAMDPSF, VELReader):
    """ This class contains methods for velocity file analysis. """

    def __init__(self, psfFile, velFile=None):

        NAMDPSF.__init__(self, psfFile)
        VELReader.__init__(self)

        if velFile:
            self.importVELFile(velFile)

        
        
    #---------------------------------------------
    #_Data accession methods
    #---------------------------------------------

    def getKineticEnergyDistribution(self, selection='all', binSize=0.1):
        """ This method can be used to compute the kinetic energy distribution without plotting it.
            This method requires a .psf to be loaded to have access to atom masses

            Input:  selection   -> atom selection from psf data
                    binSize     -> the size of the bin

            Output: numpy 2D array containing the bin mean value (first column)
                    and the corresponding density (second column) """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = parent.psfData.getSelection(selection)

        try: #_Try to get the masses list for each selected atoms
            massList = parent.psfData.getAtomsMasses(selection) 
        except AttributeError:
            print("No psf data can be found in the NAMDAnalyzer object.\n Please load a .psf file.\n")
            return

        #_Computes the kinetic energy for each atom
        data = 0.5 * massList * np.sum(self.velData**2, axis=1)[selection] 
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
                if it.value <= val:
                    density[i] += 1
                else:
                    break #_Stops the iteration if the value is larger and continue to the next bin

        #_Adding one to each entry of the density to compensate for the first missing increment, which
        #_is due to the while loop.
        density += 1

        #_Finally we normalize the density vector
        density /= ( binSize * data.size )

        return np.column_stack((xBins, density))

 

    #---------------------------------------------
    #_Plotting methods
    #---------------------------------------------

    def plotKineticEnergyDistribution(self, selection="all", binSize=0.1, fit=False, model=None, p0=None):
        """ This method calls pylab's hist method is used to plot the distribution.
            This method requires a .psf to be loaded to have access to atom masses

            Input:  binSize -> the size of the bin. Determine the width of each rectangle of the histogram 
                    begin   -> start index for data
                    end     -> end index for data
                    fit     -> if set to True, use the given model in scipy's curve_fit method and plot it
                    model   -> model to be used for the fit 
                    p0      -> starting parameter for fitting """


        dist = self.getKineticEnergyDistribution(selection, binSize)
        xBins = dist[:,0]
        data  = dist[:,1]
    
        fig, ax = plt.subplots()

        ax.plot(xBins, data)
        ax.set_xlabel(r'$Kinetic \ energy \ (kcal/mol)$')
        ax.set_ylabel('Density')

        if fit:
            if model is None:
                print("Fit model error: fit was set to 'True' but no model was given.\n")
                return

            #_Extract data directly from the plot (using the first entry in case of subplots), and call
            #_scipy's curve_fit procedure with the given model. Then plot the fitted model on top.
            params = curve_fit(model, xBins, data, p0=p0, method='trf')
            ax.plot(xBins, model(xBins, *params[0]), color='orange', label='Fit params:\n' + 
                                                            "\n".join(params[0].astype(str)))
            plt.legend(framealpha=0.5)

            plt.tight_layout()
            return plt.show(block=False)

        else:
            plt.tight_layout()
            return plt.show(block=False)

