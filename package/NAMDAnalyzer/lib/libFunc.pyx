from struct import *

import numpy as np

cimport numpy as np
cimport cython

np.import_array()


cdef extern from "libFunc.h":

    int getDCDCoor(char *fileName, int *frames, int nbrFrames, int nbrAtoms, int *selAtoms, 
                    int selAtomsSize, int *dims, int nbrDims, int cell, int *startPos, float *outArr);

    int getDCDCell(char *fileName, int *frames, int nbrFrames, int *startPos, double *outArr);


    void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                            float *donors, int size_donors,
                            float *hydrogens, int size_hydrogens, 
                            float *cellDims,
                            float *out, int maxTime, int step, int nbrTimeOri,
                            float maxR, float minAngle, int continuous);


    void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                         float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                         float *out, int out_dim0, int out_dim1, 
                         int nbrTS, int nbrTimeOri);


    void getDistances(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                      float *out, float *cellDims, int nbrFrames, int sameSel);

    void getRadialNbrDensity(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                             float *out, int outSize, float *cellDims, int nbrFrames, int sameSel, 
                             float maxR, float dr);

    void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames,
                    int *refSel, int refSize, int *outSel, int outSelSize,
                    int *out, float *cellDims, float distance);


    int getParallelBackend();



def py_getDCDCoor( fileName, 
                np.ndarray[int, ndim=1, mode="c"] frames not None, nbrAtoms,
                np.ndarray[int, ndim=1, mode="c"] selAtoms not None, 
                np.ndarray[int, ndim=1, mode="c"] dims not None, cell,
                np.ndarray[int, ndim=1, mode="c"] startPos not None, 
                np.ndarray[float, ndim=3, mode="c"] outArr not None): 

    getDCDCoor( fileName,
                <int*> np.PyArray_DATA(frames), len(frames), nbrAtoms, 
                <int*> np.PyArray_DATA(selAtoms), len(selAtoms),  
                <int*> np.PyArray_DATA(dims), len(dims), cell,
                <int*> np.PyArray_DATA(startPos),
                <float*> np.PyArray_DATA(outArr) )

def py_getDCDCell( fileName, 
                np.ndarray[int, ndim=1, mode="c"] frames not None, 
                np.ndarray[int, ndim=1, mode="c"] startPos not None, 
                np.ndarray[double, ndim=2, mode="c"] outArr not None): 

    getDCDCell( fileName,
                <int*> np.PyArray_DATA(frames), len(frames), 
                <int*> np.PyArray_DATA(startPos),
                <double*> np.PyArray_DATA(outArr) )





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



def py_getDistances( np.ndarray[float, ndim=3, mode="c"] sel1 not None,
                     np.ndarray[float, ndim=3, mode="c"] sel2 not None,
                     np.ndarray[float, ndim=2, mode="c"] out not None,
                     np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                     int sameSel):
    getDistances(<float*> np.PyArray_DATA(sel1), int(sel1.shape[0]),
                 <float*> np.PyArray_DATA(sel2), int(sel2.shape[0]), 
                 <float*> np.PyArray_DATA(out),
                 <float*> np.PyArray_DATA(cellDims),
                 cellDims.shape[0], sameSel)



def py_getRadialNbrDensity( np.ndarray[float, ndim=3, mode="c"] sel1 not None,
                     np.ndarray[float, ndim=3, mode="c"] sel2 not None,
                     np.ndarray[float, ndim=1, mode="c"] out not None,
                     np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                     int sameSel, float maxR, float dr):
    getRadialNbrDensity(<float*> np.PyArray_DATA(sel1), int(sel1.shape[0]),
                 <float*> np.PyArray_DATA(sel2), int(sel2.shape[0]), 
                 <float*> np.PyArray_DATA(out), int(out.size),
                 <float*> np.PyArray_DATA(cellDims),
                 cellDims.shape[0], sameSel, maxR, dr)




def py_getHydrogenBonds( np.ndarray[float, ndim=3, mode="c"] acceptors not None, nbrFrames,
                         np.ndarray[float, ndim=3, mode="c"] donors not None,
                         np.ndarray[float, ndim=3, mode="c"] hydrogens not None,
                         np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                         np.ndarray[float, ndim=1, mode="c"] out not None,
                         maxTime, step, nbrTimeOri, maxR, minAngle, continuous):
    getHydrogenBonds(<float*> np.PyArray_DATA(acceptors), int(acceptors.shape[0]), nbrFrames, 
                        <float*> np.PyArray_DATA(donors), int(donors.shape[0]), 
                        <float*> np.PyArray_DATA(hydrogens), int(hydrogens.shape[0]), 
                        <float*> np.PyArray_DATA(cellDims), 
                        <float*> np.PyArray_DATA(out),
                        maxTime, step, nbrTimeOri, maxR, minAngle, continuous)




def py_getWithin( np.ndarray[float, ndim=3, mode="c"] allAtoms not None,
                  np.ndarray[int, ndim=1, mode="c"] refSel not None,
                  np.ndarray[int, ndim=1, mode="c"] outSel not None,
                  np.ndarray[int, ndim=2, mode="c"] out not None,
                  np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                  float distance ):

    getWithin(<float*> np.PyArray_DATA(allAtoms), int(allAtoms.shape[0]), int(allAtoms.shape[1]),
              <int*> np.PyArray_DATA(refSel), refSel.size, 
              <int*> np.PyArray_DATA(outSel), outSel.size, 
              <int*> np.PyArray_DATA(out), <float*> np.PyArray_DATA(cellDims),  float(distance) )






@cython.boundscheck(False)
@cython.wraparound(False)
def py_cdf(np.ndarray[float, ndim=1, mode="c"] dist not None, 
           np.ndarray[float, ndim=1, mode="c"] out not None, 
           float maxR, float dr, int normFactor):
    """ Given a distances array, computes the cumulative radial distribution with bins given 
        by range(0, maxR, dr). 

    """


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


