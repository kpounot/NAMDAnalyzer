#include <cstdio>
#include <cmath>
#include <list>
#include <omp.h>

#include "../../libFunc.h"


void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames, int *refSel, int refSize, 
                int *outSel, int outSelSize, int *out, float *cellDims, float distance)
{
    /*  The function computes, for each atom in sel, the distance with all other atoms in atomPos.
     *  Then, Periodic Boundary Conditions (PBC) are applied using cell dimensions in cellDims.
     *  Finally, computed distances are compared to required 'within distance' and if lower or equal, the
     *  corresponding entry is set to 1 in the out array.
     *
     *  Input:  allAtoms    -> flattened 2D array of atom positions for a given frame
     *                         dimensions: atoms (0), coordinates (1)
     *          selAtoms    -> same as atomPos but contains only the atoms from which distance will be 
     *                         computed
     *          nbrAtoms    -> total number of atoms in the system
     *          cellDims    -> dimensions of the cell in x, y and z (used for periodic boundary conditions) 
     *          out         -> output array of floats to store within atoms, must be 1D array of zeros 
     *                         of size nbrAtoms 
     *          distance    -> distance (not squared) within which atoms should be kept */



    float squaredDist = distance*distance;

    for(int frame=0; frame < nbrFrames; ++frame)
    { 
        float cD_x = cellDims[3*frame];
        float cD_y = cellDims[3*frame + 1];
        float cD_z = cellDims[3*frame + 2];

        for(int refId=0; refId < refSize; ++refId)
        {
            int refIdx = refSel[refId];

            float sel_x = allAtoms[3*nbrFrames*refIdx + 3*frame];
            float sel_y = allAtoms[3*nbrFrames*refIdx + 3*frame + 1];
            float sel_z = allAtoms[3*nbrFrames*refIdx + 3*frame + 2];

            #pragma omp parallel for shared(frame, refId, cD_x, cD_y, cD_z, sel_x, sel_y, sel_z, refIdx)
            for(int atomId=0; atomId < outSelSize; ++atomId)
            {
                int atomIdx = outSel[atomId];

                float atom_x = allAtoms[3*nbrFrames*atomIdx + 3*frame];
                float atom_y = allAtoms[3*nbrFrames*atomIdx + 3*frame + 1];
                float atom_z = allAtoms[3*nbrFrames*atomIdx + 3*frame + 2];

                // Only executes following if this atom was not found yet
                if(out[ outSel[atomId*nbrFrames + frame ] * nbrFrames + frame ] == 0 || refIdx != atomIdx) 
                {
                    // Computes distances for given timestep and atom
                    float dist_x = atom_x - sel_x;
                    float dist_y = atom_y - sel_y;
                    float dist_z = atom_z - sel_z;
                
                    // Apply PBC conditions
                    dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
                    if(dist_x > squaredDist)
                        continue;
                    dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
                    if(dist_y > squaredDist)
                        continue;
                    dist_z = dist_z - cD_z * roundf( dist_z / cD_z );
                    if(dist_z > squaredDist)
                        continue;

                    float dr = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;

                    if(dr <= squaredDist) 
                    {
                        out[ atomIdx * nbrFrames + frame ] = 1;
                    }
                }

                else if(refIdx == atomIdx)
                {
                    out[ atomIdx * nbrFrames + frame ] = 1;
                }

            } // atoms indices loop
        } // sel loop
    } // frames loop

}

