#include "../../libFunc.h"

void cu_waterOrientAtSurface_wrapper(float *waterO, int sizeO, float *watVec, float *prot, 
                                     int sizeP, float *out, float *cellDims, int nbrFrames, 
                                     float minR, float maxR, int maxN);

void waterOrientAtSurface(float *waterO, int sizeO, float *watVec, float *prot, int sizeP, float *out, 
                          float *cellDims, int nbrFrames, float minR, float maxR, int maxN)
{
    /*  The function computes, for each water oxygen in waterO, the orientation of water dipole moment
     *  relative to protein surface. The result is stored in an array where each position corresponds
     *  to the atoms in protein selection.
     *
     *  Input:  waterO        -> coordinates for water oxygens in all selected frames
     *          sizeO         -> number of oxygen atoms in waterO
     *          watVec        -> coordinates for water dipole moment vectors corresponding to each oxygen
     *          prot          -> coordinates for selected protein atoms in all selected frames
     *          sizeP         -> number of selected atoms in prot
     *          out           -> output array, of length equal to number of protein atoms
     *          cellDims      -> cell dimensions for each selected frames
     *          nbrFrames     -> number of frames used
     *          maxR          -> maximum distance to protein surface to use to compute orientation
     *          maxN          -> maximum distance to find nearby protein atoms
     */

    cu_waterOrientAtSurface_wrapper(waterO, sizeO, watVec, prot, sizeP, out, cellDims, 
                                    nbrFrames, minR, maxR, maxN);

}

