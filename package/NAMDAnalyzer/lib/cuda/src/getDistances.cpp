#include "../../libFunc.h"


void cu_getDistances_wrapper(  float *atoms1, int atoms1_size, float *atoms2, int atoms2_size,
                                float *cellDims, float *out );

void getDistances(float *sel1, int size_sel1, float *sel2, int size_sel2, float *cellDims, float *out)
{
    cu_getDistances_wrapper(sel1, size_sel1, sel2, size_sel2, cellDims, out);
}

