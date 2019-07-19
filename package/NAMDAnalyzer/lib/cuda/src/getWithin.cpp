#include <cstdio>
#include <cmath>
#include <list>
#include "getWithin.h"


void cu_getWithin_wrapper(float *allAtoms, int nbrAtoms, float *selAtoms, int sel_dim0, float *cellDims,
                                int *out, float distance);


void getWithin(float *allAtoms, int nbrAtoms, float *selAtoms, int sel_dim0, float *cellDims,
                                                                                    int *out, float distance)
{
    cu_getWithin_wrapper(allAtoms, nbrAtoms, selAtoms, sel_dim0, cellDims, out, distance);
}

