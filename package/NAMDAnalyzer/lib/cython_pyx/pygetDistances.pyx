cimport numpy as np
cimport cython

np.import_array()


cdef extern from "getDistances.h":
    void getDistances(float *sel1, int size_sel1, float *sel2, int size_sel2, float *cellDims, float *out);


@cython.boundscheck(False)
@cython.wraparound(False)
def py_getDistances( np.ndarray[float, ndim=2, mode="c"] sel1 not None,
                     np.ndarray[float, ndim=2, mode="c"] sel2 not None,
                     np.ndarray[float, ndim=1, mode="c"] cellDims not None,
                     np.ndarray[float, ndim=2, mode="c"] out not None):
    getDistances(<float*> np.PyArray_DATA(sel1), int(sel1.shape[0]),
                 <float*> np.PyArray_DATA(sel2), int(sel2.shape[0]), 
                 <float*> np.PyArray_DATA(cellDims), 
                 <float*> np.PyArray_DATA(out) )
