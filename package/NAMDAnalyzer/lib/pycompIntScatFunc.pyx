cimport numpy as np
cimport cython

np.import_array()


cdef extern from "compIntScatFunc.h":
    void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                         float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                         double complex *out, int out_dim0, int out_dim1, 
                         int nbrBins, int minFrames, int maxFrames, int nbrTimeOri)



@cython.boundscheck(False)
@cython.wraparound(False)
def py_compIntScatFunc( np.ndarray[float, ndim=3, mode="c"] atomPos not None,
                        np.ndarray[float, ndim=3, mode="c"] qVecs not None,
                        np.ndarray[np.complex128_t, ndim=2, mode="c"] out not None,
                        int nbrBins, int minFrames, int maxFrames, int nbrTimeOri ):
    compIntScatFunc(<float*> np.PyArray_DATA(atomPos), atomPos.shape[0], atomPos.shape[1], atomPos.shape[2],
                    <float*> np.PyArray_DATA(qVecs), qVecs.shape[0], qVecs.shape[1], qVecs.shape[2],
                    <double complex*> np.PyArray_DATA(out), out.shape[0], out.shape[1],
                    nbrBins, minFrames, maxFrames, nbrTimeOri )
