import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import prange

np.import_array()


cdef extern from "libFunc.h":
    void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                            float *donors, int size_donors,
                            float *hydrogens, int size_hydrogens, 
                            float *out, int maxTime, int step, int nbrTimeOri,
                            float maxR, float minAngle, int continuous);


    void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                         float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                         float *out, int out_dim0, int out_dim1, 
                         int nbrTS, int nbrTimeOri);


    void getDistances(float *maxSel, int maxSize, float *minSel, int minSize, 
                      float *out, float *cellDims, int sameSel);


    void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames,
                    int *selAtoms, int sel_dim0,
                    int *out, float *cellDims, float distance);


    int getParallelBackend();




@cython.boundscheck(False)
@cython.wraparound(False)
def py_compIntScatFunc( np.ndarray[float, ndim=3, mode="c"] atomPos not None,
                        np.ndarray[float, ndim=3, mode="c"] qVecs not None,
                        np.ndarray[float, ndim=2, mode="c"] out not None,
                        int nbrTS, int nbrTimeOri ):

    compIntScatFunc(<float*> np.PyArray_DATA(atomPos), atomPos.shape[0], atomPos.shape[1], atomPos.shape[2],
                    <float*> np.PyArray_DATA(qVecs), qVecs.shape[0], qVecs.shape[1], qVecs.shape[2], 
                    <float*> np.PyArray_DATA(out), out.shape[0], out.shape[1],
                    nbrTS, nbrTimeOri )




@cython.boundscheck(False)
@cython.wraparound(False)
def py_getCenterOfMass(np.ndarray[float, ndim=3] allAtoms, np.ndarray[double, ndim=2] atomMasses):
    """ Computes the center of mass for all molecules in all frames and returns an array containing the 
        center of mass coordinates (dim 1) for all frames (dim0). """


    cdef int i
    cdef int nbrFrames  = allAtoms.shape[1]
    cdef np.ndarray[double, ndim=2] out = np.zeros( (allAtoms.shape[1], allAtoms.shape[2]) ) 


    for i in range(nbrFrames):
        COM = np.dot(allAtoms[:,i,:].T, atomMasses).T
        COM = np.sum(COM, axis=0) / np.sum(atomMasses)

        out[i] = COM


    return out



@cython.boundscheck(False)
@cython.wraparound(False)
def py_getDistances( np.ndarray[float, ndim=2, mode="c"] maxSel not None,
                     np.ndarray[float, ndim=2, mode="c"] minSel not None,
                     np.ndarray[float, ndim=2, mode="c"] out not None,
                     np.ndarray[float, ndim=1, mode="c"] cellDims not None,
                     int sameSel):
    getDistances(<float*> np.PyArray_DATA(maxSel), int(maxSel.shape[0]),
                 <float*> np.PyArray_DATA(minSel), int(minSel.shape[0]), 
                 <float*> np.PyArray_DATA(out),
                 <float*> np.PyArray_DATA(cellDims),
                 sameSel)



@cython.boundscheck(False)
@cython.wraparound(False)
def py_getHydrogenBonds( np.ndarray[float, ndim=3, mode="c"] acceptors not None, nbrFrames,
                         np.ndarray[float, ndim=3, mode="c"] donors not None,
                         np.ndarray[float, ndim=3, mode="c"] hydrogens not None,
                         np.ndarray[float, ndim=1, mode="c"] out not None,
                         maxTime, step, nbrTimeOri, maxR, minAngle, continuous):
    getHydrogenBonds(<float*> np.PyArray_DATA(acceptors), int(acceptors.shape[0]), nbrFrames, 
                        <float*> np.PyArray_DATA(donors), int(donors.shape[0]), 
                        <float*> np.PyArray_DATA(hydrogens), int(hydrogens.shape[0]), 
                        <float*> np.PyArray_DATA(out),
                        maxTime, step, nbrTimeOri, maxR, minAngle, continuous)




@cython.boundscheck(False)
@cython.wraparound(False)
def py_getWithin( np.ndarray[float, ndim=3, mode="c"] allAtoms not None,
                  np.ndarray[int, ndim=1, mode="c"] selAtoms not None,
                  np.ndarray[int, ndim=2, mode="c"] out not None,
                  np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                  float distance ):

    getWithin(<float*> np.PyArray_DATA(allAtoms), int(allAtoms.shape[0]), int(allAtoms.shape[1]),
              <int*> np.PyArray_DATA(selAtoms), int(selAtoms.shape[0]), 
              <int*> np.PyArray_DATA(out), <float*> np.PyArray_DATA(cellDims),  float(distance) )





@cython.boundscheck(False)
@cython.wraparound(False)
def py_setCenterOfMassAligned(np.ndarray[float, ndim=3] allAtoms, np.ndarray[double, ndim=2] COM):
    """ Computes the center of mass for all molecules in all frames and returns an array containing the 
        center of mass coordinates (dim 1) for all frames (dim0). """


    cdef int i
    cdef int nbrFrames  = allAtoms.shape[1]

    for i in range(nbrFrames):
        allAtoms[:,i] = allAtoms[:,i] - COM[i]





@cython.boundscheck(False)
@cython.wraparound(False)
def py_cdf(np.ndarray[float, ndim=1, mode="c"] dist not None, 
           np.ndarray[float, ndim=1, mode="c"] out not None, 
           float maxR, float dr, int normFactor):
    """ Given a distances array, computes the cumulative radial distribution with bins given 
        by range(0, maxR, dr). """


    cdef int d
    cdef int size_dist = dist.size
    cdef int size_out  = out.size 
    cdef float r
    cdef int rId

    for d in range(size_dist):

        r = dist[d]
        rId = int(r / dr)

        if rId < size_out:
            out[rId] += 1 / normFactor



def py_getParallelBackend():

    return getParallelBackend()
