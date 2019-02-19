cimport numpy as np
cimport cython

np.import_array()


cdef extern from "getWithin.h":
    void getWithin(float *allAtoms, int nbrAtoms, float *selAtoms, int sel_dim0, float *cellDims,
                                                                                    int *out, float distance);


@cython.boundscheck(False)
@cython.wraparound(False)
def py_getWithin( np.ndarray[float, ndim=2, mode="c"] allAtoms not None,
                  np.ndarray[float, ndim=2, mode="c"] selAtoms not None,
                  np.ndarray[float, ndim=1, mode="c"] cellDims not None,
                  np.ndarray[int, ndim=1, mode="c"] out not None,
                  float distance ):

    getWithin(<float*> np.PyArray_DATA(allAtoms), int(allAtoms.shape[0]),
              <float*> np.PyArray_DATA(selAtoms), int(selAtoms.shape[0]), 
              <float*> np.PyArray_DATA(cellDims), 
              <int *> np.PyArray_DATA(out), float(distance) )
