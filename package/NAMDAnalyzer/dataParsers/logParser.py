"""

Classes
^^^^^^^

"""

import os
import sys
import re

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy.optimize import curve_fit

from collections import namedtuple

from NAMDAnalyzer.dataParsers.logReader import LOGReader


class NAMDLOG(LOGReader):
    """ This class takes a NAMD output logfile as input.

    """

    def __init__(self, logFile=None):

        LOGReader.__init__(self)

        if logFile:
            self.importLOGFile(logFile)



# --------------------------------------------
# Data accession methods
# --------------------------------------------
    def getDataSeries(self, keywordsStr, begin=0, end=None):
        """ This method is used to extract on or several columns
            from the full dataSet.

            :arg keywordStr: keywords string
                             (example: "ELECT MISC TOTAL")
                             The full list can be obtained
                             using *etitle* attribute
            :arg begin:      first timestep used as start of data series
            :arg end:        last timestep to be used + 1

            :returns: numpy 2D array containing the selected
                      columns within given range

        """

        # Split the string given in arguments
        # (assuming space, comma or semicolon separation.)
        keywords = re.split('[\s,;]', keywordsStr)
        keywords = list(filter(None, keywords))

        # Add selected columns to a list
        dataSeries = []
        for key in keywords:
            dataSeries.append(self.logData[self.keywordsDict[key]][begin:end])

        return np.array(dataSeries).transpose()


    def getDataDistribution(self, keyword, binSize=50, begin=0, end=None):
        """ This method can be used to compute the distribution of
            a data series without plotting it.

            :arg keyword: the column to be used to compute the distribution
            :arg binSize: the size of the bin. Determine the width of each
                          rectangle of the histogram
            :arg begin:   first frame to be used
            :arg end:     last frame to be used + 1

            :returns: numpy 2D array containing the bin mean value
                      (first column) and the corresponding density
                      (second column).

        """

        data = np.sort(
            self.getDataSeries(keyword, begin=begin, end=end).ravel())

        # Defining the locations for binning
        xBins = np.arange(data.min(), data.max() + 1, binSize)

        density = np.zeros(xBins.shape[0])

        # Start accumulation in density vector,
        # first we create an iterator on data array
        it = np.nditer(data)
        # Then we loop on the bins
        for i, val in enumerate(xBins):
            # While we're not at the end of the iterator, we keep iterating
            while it.iternext():
                # Comparison of iterator value with high bin limit,
                # if lower, we increase density index by 1
                if it.value < val:
                    density[i] += 1
                else:
                    # Stops the iteration if the value is larger
                    # and continue to the next bin
                    break

        # Adding one to each entry of the density to compensate
        # for the first missing increment, which is due to the while loop.
        density += 1

        # Finally we normalize the density vector
        density /= np.sum(density)

        return np.column_stack((xBins, density))



# --------------------------------------------
# Plotting methods
# --------------------------------------------
    def plotDataSeries(self, keywordsStr, xaxis='TS', begin=0, end=None,
                       fit=False, fitIndex=0, model=None, p0=None):
        """ This method can be used to quickly plot one or several data series.

            :arg keywordStr: keywords string (example: "ELECT MISC TOTAL")
                             The full list can be obtained using *etitle*
                             attribute
            :arg xaxis:      data series to be used on x-axis
                             (default 'TS' for number of time steps)
            :arg begin:      first frame to be used
            :arg end:        last frame to be used + 1
            :arg fit:        whether data should be fitted against a
                             given model
                             (using Scipy curve_fit)
            :arg fitIndex:   if several keywords for data series are given,
                             this allows to select which data series is to be
                             fitted (default 0 for the first in the string)
            :arg model:      model to be used for fitting, will be given to
                             Scipy curve_fit
            :arg p0:         initial parameters for Scipy curve_fit

        """

        # Split the string given in arguments
        # (assuming space, comma or semicolon separation.)
        keywords = re.split('[\s,;]', keywordsStr)
        keywords = list(filter(None, keywords))

        # Get the x-axis values for the plot
        xData = self.logData[self.keywordsDict[xaxis]][begin:end]
        # If x-axis consists in timestep, convert it to time with
        # magnitude given by scale factor
        if xaxis == 'TS':
            xData = xData * self.timestep

        # Get selected data series
        dataSeries = self.getDataSeries(keywordsStr, begin=begin, end=end)

        # Initializing color scale
        norm = colors.Normalize(0, dataSeries.shape[1])
        colorList = cm.jet(norm(np.arange(dataSeries.shape[1])))

        # Create figure and plot the first column to define x-axis
        fig, ax1 = plt.subplots()
        ax1.plot(xData, dataSeries[:, 0], color=colorList[0])

        if xaxis == 'TS':
            ax1.set_xlabel('Time (s)')
        else:
            ax1.set_xlabel(xaxis)

        ax1.set_ylabel(keywords[0], color=colorList[0])
        ax1.tick_params(axis='y', labelcolor = colorList[0])

        # For each additional data series, twinx() method is used to
        # superimpose data with their own y-axis
        offset = 0.2
        for col, values in enumerate(dataSeries[0, 1:]):
            ax = ax1.twinx()
            ax.plot(xData, dataSeries[:, col + 1], color=colorList[col + 1])
            ax.set_ylabel(keywords[col + 1], color=colorList[col + 1])
            ax.tick_params(axis='y', labelcolor = colorList[col + 1])

            # Shift the y-axis to the right to avoid overlaps
            ax.spines["right"].set_position(("axes", 1 + col * offset))

        if fit:
            if model is None:
                print("Fit model error: fit was set to 'True' "
                      "but no model was given.\n")
                return

            # Extract data directly from the plot (using the first
            # entry in case of subplots), and call
            # scipy's curve_fit procedure with the given model.
            # Then plot the fitted model on top.
            params = curve_fit(model, xData, dataSeries[:, fitIndex])
            ax.plot(xData, model(xData, *params[0]),
                    label='Fit params: ' + "\n".join(params[0]))
            plt.legend(framealpha=0.5)

            fig.show()

        else:
            fig.show()


    def plotDataDistribution(self, keyword, binSize=50, begin=0, end=None,
                             fit=False, model=None, p0=None):
        """ This method takes one data series as argument, and computes the
            number occurences of each value within a range determined by the
            binSize parameter.

            :arg keyword: the column to be used to compute the distribution
            :arg binSize: the size of the bin. Determine the width of each
                          rectangle of the histogram
            :arg begin:   first frame to be used
            :arg end:     last frame to be used + 1
            :arg fit:     if set to True, use the given model in Scipy
                          curve_fit method and plot it
            :arg model:   model to be used for the fit
            :arg p0:      initial parameters for Scipy cure_fit

        """


        keyData = np.sort(
            self.getDataSeries(keyword, begin=begin, end=end).ravel())

        # Defining the locations for binning
        xBins = np.arange(keyData.min(), keyData.max() + 1, binSize)

        fig, ax = plt.subplots()

        ax.hist(keyData, bins=xBins, density=True, edgecolor='black')
        ax.set_xlabel(keyword)
        ax.set_ylabel('Density')

        if fit:
            if model is None:
                print("Fit model error: fit was set to 'True' "
                      "but no model was given.\n")
                return

            # Extract data directly from the plot (using the first entry
            # in case of subplots), and call
            # scipy's curve_fit procedure with the given model.
            # Then plot the fitted model on top.
            plotData = np.array(
                [(patch.get_x(), patch.get_height()) for patch in ax.patches])
            params = curve_fit(model, plotData[:, 0], plotData[:, 1], p0=p0)
            ax.plot(plotData[:, 0], model(plotData[:, 0], *params[0]),
                    color='orange', label='Fit params:\n '
                                          '\n'.join(params[0].astype(str)))
            fig.legend(framealpha=0.5)

            fig.show()

        else:
            fig.show()
