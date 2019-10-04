#include "../../libFunc.h"

void cu_getWithin_wrapper(float *allAtoms, int nbrAtoms, int nbrFrames, 
                            int *refSel, int refSize,
                            int *outSel, int outSelSize,
                            int *out, float *cellDims, float distance);


void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames,
                int *refSel, int refSize, int *outSel, int outSelSize,
                int *out, float *cellDims, float distance)
{
    cu_getWithin_wrapper(allAtoms, nbrAtoms, nbrFrames, refSel, refSize, 
                         outSel, outSelSize, out, cellDims, distance);
}

