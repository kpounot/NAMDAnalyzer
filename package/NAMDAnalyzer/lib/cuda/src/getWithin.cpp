#include "../../libFunc.h"

void cu_getWithin_wrapper(float *allAtoms, int nbrAtoms, int nbrFrames, 
                            int *selAtoms, int sel_dim0, float *cellDims, int *out, float distance);


void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames, 
                int *selAtoms, int sel_dim0, float *cellDims, int *out, float distance)
{
    cu_getWithin_wrapper(allAtoms, nbrAtoms, nbrFrames, selAtoms, sel_dim0, cellDims, out, distance);
}

