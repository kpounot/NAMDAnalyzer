import os, sys

import psutil

import numpy as np


def getSubArraySlices(dim1, dim2, elemSize, extraDimSize=1):
    """ This function takes two dimensions of a 2D matrix, as well as the size in bytes of elements.
        Then, the amount of data needed to store the matrix is compared to currently available memory.

        If the 2D matrix can fit, the initial dimensions are simply returned in a list of tuple containing
        slices for all rows and columns.
        Otherwise, a list of tuples of slices is returned containing slices for each sub matrix. 
        
        An extraDimSize parameter can be provided in case of 3D array, such that the third dimension can
        be taken into account - used for example for temperory storage of vector coordinates in a third
        dimension, that will disappear after scalar product. """


    size = dim1 * dim2 * elemSize
    
    avail_mem = psutil.virtual_memory().available

    
    if size < avail_mem:
        return [ (slice(0, None, 1), slice(0, None, 1)) ] 

    else:
        newDim1 = dim1
        newDim2 = dim2

        maxDims = int(avail_mem / elemSize)

        while(dim1*dim2 > maxDims / extraDimSize):
            newDim1 *= 0.95
            newDim2 *= 0.95

        x_idxList = np.arange(0, dim1, newDim1) 
        y_idxList = np.arange(0, dim2, newDim2) 

        sliceList = []
        for xIdx in x_idxList:
            for yIdx in y_idxList:
                sliceList.append( (slice(xIdx, xIdx+newDim1, 1), slice(yIdx, yIdx+newDim2, 1)) )


        return sliceList
