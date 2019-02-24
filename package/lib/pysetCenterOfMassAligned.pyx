import numpy as np

cimport numpy as np
cimport cython

np.import_array()


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def py_setCenterOfMassAligned(np.ndarray[float, ndim=3] allAtoms, np.ndarray[double, ndim=2] COM):
    """ Computes the center of mass for all molecules in all frames and returns an array containing the 
        center of mass coordinates (dim 1) for all frames (dim0). """


    cdef int i
    cdef int nbrFrames  = allAtoms.shape[1]

    for i in range(nbrFrames):
        allAtoms[:,i] = allAtoms[:,i] - COM[i]

