cimport numpy as np
cimport cython



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getCenterOfMass(np.ndarray[np.float32, ndim=3] allAtoms, np.ndarray[np.float32, ndim=3] atomMasses):
    """ Computes the center of mass for all molecules in all frames and returns an array containing the 
        center of mass coordinates (dim 1) for all frames (dim0). """


    cdef int i
    cdef int nbrFrames  = allAtoms.shape[1]
    cdef np.ndarray[np.float32, ndim=2] out = np.zeros( (allAtoms.shape[1], allAtoms.shape[2]) ) 


    for i in range(nbrFrames):
        COM = np.dot(allAtoms[:,i,:].T, atomMasses).T
        out[:,i] = COM


    return out
