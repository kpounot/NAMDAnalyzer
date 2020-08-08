"""

Classes
^^^^^^^

"""

import os
import sys
import numpy as np
import re

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from NAMDAnalyzer.dataParsers.velReader import VELReader


class NAMDVEL(VELReader):
    """ This class contains methods for velocity file analysis.

        :arg parent:  a parent class, usually a :class:`Dataset` instance
                      It is used to compute kinetic energy for instance,
                      by extracting atom masses for a loaded .psf file
                      located in :class:`Dataset`.
        :arg velFile: the velocity file to be loaded

    """

    def __init__(self, parent, velFile=None):

        self.parent = parent

        VELReader.__init__(self)

        if velFile:
            self.importVELFile(velFile)



# --------------------------------------------
# Data accession methods
# --------------------------------------------

    def getKineticEnergyDistribution(self, selection='all', binSize=0.1):
        """ This method can be used to compute the kinetic energy distribution
            without plotting it. This method requires a .psf to be loaded to
            have access to atom masses

            :arg selection: atom selection from psf data
            :arg binSize:   the size of the bin

            :returns: numpy 2D array containing the bin mean value
                      (first column) and the corresponding density
                      (second column)

        """

        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.parent.selection(selection)

        try:  # Try to get the masses list for each selected atoms
            massList = selection.getMasses()
        except AttributeError:
            print("No psf data can be found in the NAMDAnalyzer object.\n "
                  "Please load a .psf file.\n")
            return

        # Computes the kinetic energy for each atom
        data = 0.5 * massList * np.sum(self.velData**2, axis=1)[selection]
        data = np.sort(data)

        # Defining the locations for binning
        xBins = np.arange(data.min(), data.max() + 1, binSize)

        density = np.zeros(xBins.shape[0])

        it = np.nditer(data)
        for i, val in enumerate(xBins):
            while it.iternext():
                if it.value <= val:
                    density[i] += 1
                else:
                    break

        # Adding one to each entry of the density to compensate for the
        # first missing increment, which is due to the while loop.
        density += 1

        # Finally we normalize the density vector
        density /= (binSize * data.size)

        return np.column_stack((xBins, density))



# --------------------------------------------
# Plotting methods
# --------------------------------------------

    def plotKineticEnergyDistribution(self, selection="all", binSize=0.1,
                                      fit=False, model=None, p0=None):
        """ This method calls pylab's hist method is used to
            plot the distribution. This method requires a .psf to be
            loaded to have access to atom masses

            :arg binSize: the size of the bin. Determine the width of each
                          rectangle of the histogram
            :arg fit:     if set to True, use the given model in Scipy
                          curve_fit method and plot it
            :arg model:   model to be used for the fit, using Scipy curve_fit
            :arg p0:      starting parameter(s) for fitting

        """


        dist = self.getKineticEnergyDistribution(selection, binSize)
        xBins = dist[:, 0]
        data  = dist[:, 1]

        fig, ax = plt.subplots()

        ax.plot(xBins, data)
        ax.set_xlabel(r'$Kinetic \ energy \ (kcal/mol)$')
        ax.set_ylabel('Density')

        if fit:
            if model is None:
                print("Fit model error: fit was set to 'True' but "
                      "no model was given.\n")
                return

            # Extract data directly from the plot (using the first entry
            # in case of subplots), and call scipy's curve_fit procedure
            # with the given model. Then plot the fitted model on top.
            params = curve_fit(model, xBins, data, p0=p0, method='trf')
            ax.plot(xBins, model(xBins, *params[0]),
                    color='orange',
                    label='Fit params:\n\n'.join(params[0].astype(str)))
            plt.legend(framealpha=0.5)

            plt.tight_layout()
            return plt.show(block=False)

        else:
            plt.tight_layout()
            return plt.show(block=False)
