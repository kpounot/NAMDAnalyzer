import os, sys
import numpy as np
import IPython

from . import Dataset

class NAMDAnalyzer:
    """ Main class for NAMDAnalyzer.
        Contains a list of dataset, methods to create or delete them. """

    def __init__(self, fileList=None):

        self.dataSetList = []


        if fileList:
            self.newDataset(fileList)


    def newDataset(self, fileList=None):
        """ Appends a new Dataset instance to self.dataSetList and initialize it with the given
            file list. """

        self.dataSetList.append(Dataset.Dataset(fileList))



    def delDataset(self, index):
        """ Delete the dataset at the given index in self.DataSetList. """

        self.dataSetList.pop(index)



if __name__ == '__main__':

    fileList = sys.argv[1:]

    data = NAMDAnalyzer(fileList)

        
