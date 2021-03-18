from struct import *

import numpy as np

cimport numpy as np
cimport cython

np.import_array()


cdef extern from "libFunc.h":

    int getDCDCoor(char *fileName, long *frames, int nbrFrames, long nbrAtoms, long *selAtoms, 
                    int selAtomsSize, long *dims, int nbrDims, int cell, long long *startPos, 
                    float *outArr, char byteorder);

    int getDCDCell(char *fileName, int *frames, int nbrFrames, long long *startPos,
                   double *outArr, char byteorder);


    void getHBCorr(float *acceptors, int size_acceptors, int nbrFrames,
                            float *donors, int size_donors,
                            float *hydrogens, int size_hydrogens, 
                            float *cellDims,
                            float *out, int maxTime, int step, int nbrTimeOri,
                            float maxR, float minAngle, int continuous);


    void getHBNbr(  float *acceptors, int size_acceptors, int nbrFrames,
                    float *donors, int size_donors,
                    float *hydrogens, int size_hydrogens, 
                    float *cellDims, float *out, 
                    float maxR, float minAngle );


    void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                         float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                         float *out, int out_dim0, int out_dim1, 
                         int nbrTS, int nbrTimeOri, float *scatLength);


    void getDistances(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                      float *out, float *cellDims, int nbrFrames, int sameSel);

    void getRadialNbrDensity(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                             float *out, int outSize, float *cellDims, int nbrFrames, int sameSel, 
                             float maxR, float dr);

    void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames,
                    int *refSel, int refSize, int *outSel, int outSelSize,
                    int *out, float *cellDims, float distance);

    void waterOrientAtSurface(float *waterO, int sizeO, float *watVec, float *prot, int sizeP, float *out, 
                              float *cellDims, int nbrFrames, float minR, float maxR, int maxN);

    void setWaterDistPBC(float *water, int sizeW, float *prot, int sizeP, float *cellDims, int nbrFrames,
                         int nbrWAtoms);

    int getParallelBackend();



def py_getDCDCoor( fileName, 
                np.ndarray[long, ndim=1, mode="c"] frames not None, nbrAtoms,
                np.ndarray[long, ndim=1, mode="c"] selAtoms not None, 
                np.ndarray[long, ndim=1, mode="c"] dims not None, cell,
                np.ndarray[long long, ndim=1, mode="c"] startPos not None, 
                np.ndarray[float, ndim=3, mode="c"] outArr not None, byteorder): 

    res = getDCDCoor( fileName,
                <long*> np.PyArray_DATA(frames), len(frames), nbrAtoms, 
                <long*> np.PyArray_DATA(selAtoms), len(selAtoms),  
                <long*> np.PyArray_DATA(dims), len(dims), cell,
                <long long*> np.PyArray_DATA(startPos),
                <float*> np.PyArray_DATA(outArr),
                byteorder )

    return res



def py_getDCDCell( fileName, 
                np.ndarray[int, ndim=1, mode="c"] frames not None, 
                np.ndarray[long long, ndim=1, mode="c"] startPos not None, 
                np.ndarray[double, ndim=2, mode="c"] outArr not None,
                byteorder ): 

    res = getDCDCell( fileName,
                <int*> np.PyArray_DATA(frames), len(frames), 
                <long long*> np.PyArray_DATA(startPos),
                <double*> np.PyArray_DATA(outArr),
                byteorder )


    return res




def py_compIntScatFunc( np.ndarray[float, ndim=3, mode="c"] atomPos not None,
                        np.ndarray[float, ndim=3, mode="c"] qVecs not None,
                        np.ndarray[float, ndim=2, mode="c"] out not None,
                        int nbrTS, int nbrTimeOri,
                        np.ndarray[float, ndim=1, mode="c"] scatLength not None):

    compIntScatFunc(<float*> np.PyArray_DATA(atomPos), atomPos.shape[0], atomPos.shape[1], atomPos.shape[2],
                    <float*> np.PyArray_DATA(qVecs), qVecs.shape[0], qVecs.shape[1], qVecs.shape[2], 
                    <float*> np.PyArray_DATA(out), out.shape[0], out.shape[1],
                    nbrTS, nbrTimeOri, <float*> np.PyArray_DATA(scatLength) )





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




def py_getHBCorr( np.ndarray[float, ndim=3, mode="c"] acceptors not None, nbrFrames,
                         np.ndarray[float, ndim=3, mode="c"] donors not None,
                         np.ndarray[float, ndim=3, mode="c"] hydrogens not None,
                         np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                         np.ndarray[float, ndim=1, mode="c"] out not None,
                         maxTime, step, nbrTimeOri, maxR, minAngle, continuous):
    getHBCorr(<float*> np.PyArray_DATA(acceptors), int(acceptors.shape[0]), nbrFrames, 
                        <float*> np.PyArray_DATA(donors), int(donors.shape[0]), 
                        <float*> np.PyArray_DATA(hydrogens), int(hydrogens.shape[0]), 
                        <float*> np.PyArray_DATA(cellDims), 
                        <float*> np.PyArray_DATA(out),
                        maxTime, step, nbrTimeOri, maxR, minAngle, continuous)




def py_getHBNbr( np.ndarray[float, ndim=3, mode="c"] acceptors not None, nbrFrames,
                         np.ndarray[float, ndim=3, mode="c"] donors not None,
                         np.ndarray[float, ndim=3, mode="c"] hydrogens not None,
                         np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                         np.ndarray[float, ndim=1, mode="c"] out not None,
                         maxR, minAngle):
    getHBNbr(<float*> np.PyArray_DATA(acceptors), int(acceptors.shape[0]), nbrFrames, 
                        <float*> np.PyArray_DATA(donors), int(donors.shape[0]), 
                        <float*> np.PyArray_DATA(hydrogens), int(hydrogens.shape[0]), 
                        <float*> np.PyArray_DATA(cellDims), 
                        <float*> np.PyArray_DATA(out), maxR, minAngle)




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



def py_waterOrientAtSurface( np.ndarray[float, ndim=3, mode="c"] waterO not None,
                             np.ndarray[float, ndim=3, mode="c"] watVec not None,
                             np.ndarray[float, ndim=3, mode="c"] prot not None,
                             np.ndarray[float, ndim=2, mode="c"] out not None,
                             np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                             minR, maxR, maxN ):

    waterOrientAtSurface(<float*> np.PyArray_DATA(waterO), waterO.shape[0], 
                         <float*> np.PyArray_DATA(watVec), 
                         <float*> np.PyArray_DATA(prot), prot.shape[0], 
                         <float*> np.PyArray_DATA(out), 
                         <float*> np.PyArray_DATA(cellDims), 
                         waterO.shape[1], minR, maxR, maxN);



def py_setWaterDistPBC( np.ndarray[float, ndim=3, mode="c"] water not None,
                             np.ndarray[float, ndim=3, mode="c"] prot not None,
                             np.ndarray[float, ndim=2, mode="c"] cellDims not None,
                             nbrWAtoms):

    setWaterDistPBC(<float*> np.PyArray_DATA(water), int(water.shape[0] / nbrWAtoms), 
                    <float*> np.PyArray_DATA(prot), prot.shape[0], 
                    <float*> np.PyArray_DATA(cellDims), water.shape[1], nbrWAtoms);




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



@cython.boundscheck(False)
@cython.wraparound(False)
def py_getWaterOrientVolMap(np.ndarray[int, ndim=3, mode="c"] indices not None,
                            np.ndarray[float, ndim=2, mode="c"] orientations not None, 
                            np.ndarray[int, ndim=2, mode="c"] toKeep not None, 
                            np.ndarray[float, ndim=3, mode="c"] out not None):
    """ Used by the Rotations.WaterOrientAtSurface class to compute volumetric map. """


    cdef int at
    cdef int fr

    cdef int xId
    cdef int yId
    cdef int zId

    cdef int nbrAtoms   = indices.shape[0]
    cdef int nbrFrames  = indices.shape[1] 

    for at in range(nbrAtoms):
        for fr in range(nbrFrames):

            xId = indices[at,fr,0]
            yId = indices[at,fr,1]
            zId = indices[at,fr,2]

            if(toKeep[at,fr] == 1):
                out[xId,yId,zId] += orientations[at,fr]



@cython.boundscheck(False)
@cython.wraparound(False)
def py_waterOrientHist(np.ndarray[float, ndim=1, mode="c"] orientations not None, 
                         np.ndarray[float, ndim=1, mode="c"] out not None, 
                         float nbrBins):
    """ Given an orientation array, computes the histogram in the range [-1, 1]. """


    cdef int o
    cdef int size_orient = orientations.size
    cdef int size_out    = out.size 
    cdef float angle
    cdef int aId

    cdef float dTheta = 2 / nbrBins

    for o in range(size_orient):

        angle = orientations[o]
        aId = int( (angle+1) / (dTheta*1.0001) )

        out[aId] += 1



@cython.boundscheck(False)
@cython.wraparound(False)
def py_waterNumberDensityHist(np.ndarray[float, ndim=1, mode="c"] density not None, 
                              np.ndarray[float, ndim=1, mode="c"] edges not None, 
                              np.ndarray[float, ndim=1, mode="c"] out not None ):
    """ Given an water number volumetric map, computes the histogram given provided edges. """


    cdef int d
    cdef float val

    cdef int edgeId

    cdef int size_density = density.size
    cdef int size_edges   = edges.size

    for d in range(size_density):
        val = density[d]

        for edgeId in range(size_edges - 1):
            if val >= edges[edgeId] and val < edges[edgeId + 1]:
                out[edgeId] += 1




@cython.boundscheck(False)
@cython.wraparound(False)
def py_getWaterDensityVolMap(np.ndarray[int, ndim=3, mode="c"] indices not None,
                             np.ndarray[float, ndim=3, mode="c"] out not None ):
    """ Used by the Rotations.WaterOrientAtSurface class to compute volumetric map. """


    cdef int at
    cdef int fr

    cdef int xId
    cdef int yId
    cdef int zId

    cdef int nbrAtoms  = indices.shape[0]
    cdef int nbrFrames = indices.shape[1]

    for at in range(nbrAtoms):
        for fr in range(nbrFrames):
            xId = indices[at,fr,0]
            yId = indices[at,fr,1]
            zId = indices[at,fr,2]

            out[xId,yId,zId] += 1 / nbrFrames 





def py_getParallelBackend():

    return getParallelBackend()


