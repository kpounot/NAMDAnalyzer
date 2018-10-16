import os, sys
import re

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy.optimize import curve_fit

from collections import namedtuple

class NAMDLog:
    """ This class takes a NAMD output logfile as input. """

    def __init__(self, logFile, parent=None):

        self.parent = parent

        #_Open the file and get the lines
        try:
            with open(logFile, 'r') as fileToRead:
                raw_data = fileToRead.read().splitlines()

        except UnicodeError:
            with open(logFile, 'r', encoding='utf-16') as fileToRead:
                raw_data = fileToRead.read().splitlines()

        except:    
            print("Error while reading the file.\nPlease check the file path given in argument.")
            return 

        #_Store each lines corresponding to energies output to a list
        entries = []
        for line in raw_data:
            if re.search('ENERGY:', line):
                entries.append(line.split()[1:])
            elif re.search('Info: TIMESTEP', line):
                self.timestep = float(line.split()[2])
            elif re.search('ETITLE:', line):
                self.etitle = line.split()[1:]


        #_Create a namedtuple so that data are stored in a secured way and can be retrieved using keywords
        dataTuple = namedtuple('dataTuple', " ".join(self.etitle)) 

        #_This dictionary is meant to be used to easily retrieve data series and column labels
        self.keywordsDict = {}
        for i, title in enumerate(self.etitle):
            self.keywordsDict[title] = i   

        
        #_Convert the python list to numpy array for easier manipulation
        entries = np.array(entries).astype(float)

        #_Store the data in the namedtuple according to their column/keyword
        self.dataSet = dataTuple(*[ entries[:,col] for col, val in enumerate(entries[0]) ]) 


    #---------------------------------------------
    #_Data accession methods
    #---------------------------------------------
    def getDataSeries(self, keywordsStr, begin=0, end=None):
        """ This method is used to extract on or several columns from the full dataSet.

            Input:  keywords string (example: "ELECT MISC TOTAL")
                    begin   -> first timestep used as start of data series
                    end     -> last timestep to be used + 1 

            Output: numpy 2D array containing the selected columns within given range """

        #_Split the string given in arguments (assuming space, comma or semicolon separation.)
        keywords = re.split('[\s,;]', keywordsStr)
        keywords = list(filter(None, keywords))

        #_Add selected columns to a list
        dataSeries = []
        for key in keywords:
            dataSeries.append( self.dataSet[self.keywordsDict[key]] )

        return np.array(dataSeries).transpose()[begin:end]

    
    def getDataDistribution(self, keyword, binSize=50, begin=0, end=None):
        """ This method can be used to compute the distribution of a data series without plotting it.

            Input:  keyword -> the column to be used to compute the distribution
                    binSize -> the size of the bin. Determine the width of each rectangle of the histogram 
                    begin   -> first frame to be used
                    end     -> last frame to be used + 1
                    fit     -> if set to True, use the given model in scipy's curve_fit method and plot it
                    model   -> model to be used for the fit 

            Output: numpy 2D array containing the bin mean value (first column)
                    and the corresponding density (second column) """

        data = np.sort(self.getDataSeries(keyword, begin=begin, end=end).ravel())

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
    def plotDataSeries(self, keywordsStr, xaxis='TS', timeScale=0.001, timeUnit='ps', 
                        begin=0, end=None, fit=False, fitIndex=0, model=None, p0=None):
        """ This method can be used to quickly plot one or several data series.
            
            Input: keywords string (example: "ELECT MISC TOTAL") """

        #_Split the string given in arguments (assuming space, comma or semicolon separation.)
        keywords = re.split('[\s,;]', keywordsStr)
        keywords = list(filter(None, keywords))

        #_Get the x-axis values for the plot
        xData = self.dataSet[self.keywordsDict[xaxis]][begin:end]
        #_If x-axis consists in timestep, convert it to time with magnitude given by scale factor
        if xaxis == 'TS':
            xData = xData * self.timestep * timeScale

        #_Get selected data series
        dataSeries = self.getDataSeries(keywordsStr, begin=begin, end=end)

        #_Initializing color scale
        norm = colors.Normalize(0, dataSeries.shape[1])
        colorList = cm.jet( norm( np.arange(dataSeries.shape[1]) ))

        #_Create figure and plot the first column to define x-axis
        fig, ax1 = plt.subplots()
        ax1.plot(xData, dataSeries[:,0], color=colorList[0])

        if xaxis == 'TS':
            ax1.set_xlabel('Time (' + timeUnit + ')')
        else:
            ax1.set_xlabel(xaxis)

        ax1.set_ylabel(keywords[0], color=colorList[0])
        ax1.tick_params(axis='y', labelcolor = colorList[0])

        #_For each additional data series, twinx() method is used to superimpose data with their own y-axis
        offset=0.2
        for col, values in enumerate(dataSeries[0,1:]):
            ax = ax1.twinx()
            ax.plot(xData, dataSeries[:,col+1], color=colorList[col+1])
            ax.set_ylabel(keywords[col+1], color=colorList[col+1])
            ax.tick_params(axis='y', labelcolor = colorList[col+1])

            #_Shift the y-axis to the right to avoid overlaps
            ax.spines["right"].set_position(("axes", 1 + col*offset))

        if fit:
            if model is None:
                print("Fit model error: fit was set to 'True' but no model was given.\n")
                return

            #_Extract data directly from the plot (using the first entry in case of subplots), and call
            #_scipy's curve_fit procedure with the given model. Then plot the fitted model on top.
            params = curve_fit(model, xData, dataSeries[:,fitIndex])
            ax.plot(xData, model(xData, *params[0]), label='Fit params: ' + "\n".join(params[0]))
            plt.legend(framealpha=0.5)

            plt.tight_layout()
            plt.show(block=False)

        else:
            plt.tight_layout()
            plt.show(block=False)


    def plotDataDistribution(self, keyword, binSize=50, begin=0, end=None, fit=False, model=None, p0=None):
        """ This method takes one data series as argument, and computes the number occurences of 
            each value within a range determined by the binSize parameter.

            Input:  keyword -> the column to be used to compute the distribution
                    binSize -> the size of the bin. Determine the width of each rectangle of the histogram 
                    begin   -> first frame to be used
                    end     -> last frame to be used + 1
                    fit     -> if set to True, use the given model in scipy's curve_fit method and plot it
                    model   -> model to be used for the fit """

        keyData = np.sort(self.getDataSeries(keyword, begin=begin, end=end).ravel())

        #_Defining the locations for binning
        xBins = np.arange(keyData.min(), keyData.max() + 1, binSize)

        fig, ax = plt.subplots()

        ax.hist(keyData, bins=xBins, normed=True, edgecolor='black')
        ax.set_xlabel(keyword)
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
            plt.show(block=False)

        else:
            plt.tight_layout()
            plt.show(block=False)
