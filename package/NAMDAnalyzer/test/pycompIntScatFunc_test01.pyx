import numpy as np

cimport numpy as np
cimport cython

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def py_compIntScatFunc( np.ndarray[float, ndim=3] atomPos not None,
                        np.ndarray[float, ndim=3] qVecs not None,
                        np.ndarray[np.complex128_t, ndim=2] out not None,
                        int nbrBins, int minFrames, int maxFrames, int nbrTimeOri ):

    cdef int qIdx
    cdef int binIdx
    cdef double corr
    cdef int nbrTimeOri = 25

    cdef int nbrFrames
    cdef int incr
    cdef np.ndarray[float, ndim=3] displacement = np.zeros(atomPos.shape[0], nbrTimeOri, 3)
    cdef np.ndarray[float, ndim=3] temp = np.zeros(atomPos.shape[0], nbrTimeOri, qVecs.shape[1])
    
    for qIdx in range(qVecs.shape[0]):

        for binIdx in range(nbrBins):
            print("Computing bin: %i/%i" % (binIdx+1, nbrBins), end='\r')
            nbrFrames   = minFrames + int(binIdx * (maxFrames - minFrames) / nbrBins )

            #_Defines the number of time origins to be averaged on
            #_Speeds up computation and helps to prevent MemoryError for large arrays
            incr = int(atomPos.shape[1] / nbrTimeOri) 

            #_Computes intermediate scattering function for one timestep, averaged over time origins
            displacement = atomPos[:,nbrFrames::incr] - atomPos[:,:-nbrFrames:incr]

            temp = 1j * np.dot( displacement, qVecs[qIdx] )
            np.exp( temp, out=temp )

            corr = np.mean(temp) #_Average over time origins, q vectors and atoms

            out[qIdx,binIdx] = temp


