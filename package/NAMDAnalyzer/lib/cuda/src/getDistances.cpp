#include "../../libFunc.h"

void cu_getDistances_wrapper(float *maxSel, int maxSize, float *minSel, 
                             int minSize, float *out, float *cellDims, int sameSel);

void getDistances(float *maxSel, int maxSize, float *minSel, 
                  int minSize, float *out, float *cellDims, int sameSel)
{
    /*  The function computes, for each atom in sel, the distance with all other atoms in atomPos.
     *  Then, Periodic Boundary Conditions (PBC) are applied using cell dimensions in cellDims.
     *
     *  Input:  maxSel      -> first selection of atoms coordinates (biggest dimension)
     *          maxSize     -> size of maxSel list, in number of elements
     *          minSel      -> second selection of atom coordinates, distances with sel1 will be computed
     *          minSize     -> size of minSel list, in number of elements
     *          out         -> output matrix to store the result */

    cu_getDistances_wrapper(maxSel, maxSize, minSel, minSize, out, cellDims, sameSel);

}

