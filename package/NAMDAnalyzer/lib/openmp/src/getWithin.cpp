#include <cstdio>
#include <cmath>
#include <list>
#include <omp.h>

#include "../../libFunc.h"


void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames, int *selAtoms, 
                int sel_dim0, int *out, float *cellDims, float distance)
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

    int frame, atomId;
    
    for(frame=0; frame < nbrFrames; ++frame)
    { 
        for(int i=0; i < sel_dim0; ++i)
            out[ selAtoms[i] * nbrFrames + frame ] = 1;

        float cD_x = cellDims[3*frame];
        float cD_y = cellDims[3*frame + 1];
        float cD_z = cellDims[3*frame + 2];

        #pragma omp parallel for private(atomId)
        for(int selId=0; selId < sel_dim0; ++selId)
        {
            float sel_x = allAtoms[3*nbrFrames*selAtoms[selId] + 3*frame];
            float sel_y = allAtoms[3*nbrFrames*selAtoms[selId] + 3*frame + 1];
            float sel_z = allAtoms[3*nbrFrames*selAtoms[selId] + 3*frame + 2];

            for(atomId=0; atomId < nbrAtoms; ++atomId)
            {
                float atom_x = allAtoms[3*nbrFrames*atomId + 3*frame];
                float atom_y = allAtoms[3*nbrFrames*atomId + 3*frame + 1];
                float atom_z = allAtoms[3*nbrFrames*atomId + 3*frame + 2];

                // Only executes following if this atom was not found yet
                if(out[ atomId * nbrFrames + frame ] == 0) 
                {
                    // Computes distances for given timestep and atom
                    float dist_x = atom_x - sel_x;
                    float dist_y = atom_y - sel_y;
                    float dist_z = atom_z - sel_z;
                
                    // Apply PBC conditions
                    dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
                    if(dist_x > distance)
                        continue;

                    dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
                    if(dist_y > distance)
                        continue;

                    dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

                    float dr = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;

                    if(dr <= squaredDist) 
                    {
                        #pragma omp critical
                        out[ atomId * nbrFrames + frame ] = 1;
                        break;
                    }
                }

            } // atoms indices loop
        } // sel loop
    } // frames loop

}

