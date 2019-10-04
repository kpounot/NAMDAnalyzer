#include "../../libFunc.h"

void cu_getDistances_wrapper(float *sel1, int sel1_size, float *sel2, 
                             int sel2_size, float *out, float *cellDims, int nbrFrames, int sameSel);

void getDistances(float *sel1, int sel1_size, float *sel2, 
                  int sel2_size, float *out, float *cellDims, int nbrFrames, int sameSel)
{
    /*  The function computes, for each atom in sel, the distance with all other atoms in atomPos.
     *  Then, Periodic Boundary Conditions (PBC) are applied using cell dimensions in cellDims.
     *
     *  Input:  sel1          -> first selection of atoms coordinates (biggest dimension)
     *          sel1_size     -> size of sel1 list, in number of elements
     *          sel2          -> second selection of atom coordinates, distances with sel1 will be computed
     *          sel2_size     -> size of sel2 list, in number of elements
     *          out           -> output matrix to store the result */

    cu_getDistances_wrapper(sel1, sel1_size, sel2, sel2_size, out, cellDims, nbrFrames, sameSel);

}

