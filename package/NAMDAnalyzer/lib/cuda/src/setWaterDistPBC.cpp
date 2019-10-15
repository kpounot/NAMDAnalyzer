#include "../../libFunc.h"

void cu_setWaterDistPBC_wrapper(float *water, int sizeW, float *prot, int sizeP, 
                                float *cellDims, int nbrFrames, int nbrWAtoms);

void setWaterDistPBC(float *water, int sizeW, float *prot, int sizeP, float *cellDims, 
                     int nbrFrames, int nbrWAtoms)
{
    /*  The function set periodic boundary conditions based on distances between water oxygens 
     *  and corresponding closest protein atoms.
     *
     *  Input:  waterO        -> coordinates for water oxygens in all selected frames
     *          sizeO         -> number of oxygen atoms in waterO
     *          prot          -> coordinates for selected protein atoms in all selected frames
     *          sizeP         -> number of selected atoms in prot
     *          cellDims      -> cell dimensions for each selected frames
     *          nbrFrames     -> number of frames used
     */

    cu_setWaterDistPBC_wrapper(water, sizeW, prot, sizeP, cellDims, nbrFrames, nbrWAtoms);

}

