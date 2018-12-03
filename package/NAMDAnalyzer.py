import os, sys
import numpy as np
import IPython

from . import Dataset

class NAMDAnalyzer:
    """ Main class for NAMDAnalyzer.
        Contains a list of dataset, methods to create or delete them.
        Special methods are also provided to compute quantities for several MD runs. """


    def __init__(self, fileList=None):

        self.dataSetList = []
        self.selList = dict()


        if fileList:
            self.newDataset(fileList)


    def newDataset(self, fileList=None):
        """ Appends a new Dataset instance to self.dataSetList and initialize it with the given
            file list. """

        self.dataSetList.append(Dataset.Dataset(fileList))



    def delDataset(self, index):
        """ Delete the dataset at the given index in self.DataSetList. """

        self.dataSetList.pop(index)



    def newSelection(self, selName, selText="all", segName=None, NOTsegName=None, resNbr=None, NOTresNbr=None,
                        resName=None, NOTresName=None, atom=None, NOTatom=None, index=None):
        """ Calls the self.dataSetList[0].psfData.getSelection method and store the list of selected indices 
            in the self.selList attribute.
            This method is located in the main class, therefore, it assumes that the psf file is the same
            for all dataset in self.dataSetList.
            For selections specific to a dataset, the same methods are available in Dataset class. """

        self.selList[selName] = self.dataSetList[0].psfData.getSelection(selText, segName, NOTsegName, 
                    resNbr, NOTresNbr, resName, NOTresName, atom, NOTatom, index) 


    def delSelection(self, selName):
        """ Remove the selection from self.selList at the given index. """

        self.selList.pop(selName)


              

if __name__ == '__main__':

    fileList = sys.argv[1:]

    data = NAMDAnalyzer(fileList)

        
