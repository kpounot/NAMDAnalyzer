from __future__ import print_function

import numpy as np

cimport numpy as np
cimport cython


np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def py_compIntScatFunc( np.ndarray[float, ndim=3] atomPos not None,
                        np.ndarray[float, ndim=3] qVecs not None,
                        np.ndarray[np.complex64_t, ndim=2] out not None,
                        int nbrBins, int minFrames, int maxFrames, int nbrTimeOri ):

    cdef int qIdx
    cdef int binIdx
    cdef np.complex64_t corr

    cdef int nbrFrames
    cdef int incr
    cdef np.ndarray[float, ndim=3] displacement
    cdef np.ndarray[np.complex64_t, ndim=3] temp 
    
    for qIdx in range(qVecs.shape[0]):

        print("Computing q index: %i/%i" % (qIdx+1, qVecs.shape[0]), end='\r')

        for binIdx in range(nbrBins):
            nbrFrames   = minFrames + int(binIdx * (maxFrames - minFrames) / nbrBins )

            #_Defines the number of time origins to be averaged on
            #_Speeds up computation and helps to prevent MemoryError for large arrays
            incr = int((atomPos.shape[1] - nbrFrames) / nbrTimeOri) 

            #_Computes intermediate scattering function for one timestep, averaged over time origins
            displacement = atomPos[:,nbrFrames::incr] - atomPos[:,:atomPos.shape[1]-nbrFrames:incr]

            temp = 1j * np.dot( displacement, qVecs[qIdx].T )
            np.exp( temp, out=temp )

            corr = np.mean(temp) #_Average over time origins, q vectors and atoms

            out[qIdx,binIdx] = corr


