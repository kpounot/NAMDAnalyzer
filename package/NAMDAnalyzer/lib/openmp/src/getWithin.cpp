#include <cstdio>
#include <cmath>
#include <list>
#include <omp.h>

#include "../../libFunc.h"


void getWithin(float *allAtoms, int nbrAtoms, float *selAtoms, int sel_dim0, float *cellDims,
                                                                                    int *out, float distance)
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


    for(int selAtom=0; selAtom < sel_dim0; ++selAtom)
    {

        #pragma omp parallel for
        for(int atomId=0; atomId < nbrAtoms; ++atomId)
        {

            if(out[atomId] == 0) // Only executes following if this atom was not found yet
            {
                // Computes distances for given timestep and atom
                float dist_0 = allAtoms[3 * atomId] - selAtoms[3*selAtom];
                float dist_1 = allAtoms[3 * atomId + 1] - selAtoms[3*selAtom+1];
                float dist_2 = allAtoms[3 * atomId + 2] - selAtoms[3*selAtom+2];

            
                // Applying PBC corrections
                dist_0 = abs(dist_0) - cellDims[0] * round( abs(dist_0) / cellDims[0] );
                dist_1 = abs(dist_1) - cellDims[1] * round( abs(dist_1) / cellDims[1] );
                dist_2 = abs(dist_2) - cellDims[2] * round( abs(dist_2) / cellDims[2] );


                if(dist_0 > distance || dist_1 > distance || dist_2 > distance) 
                    continue;


                float dr = dist_0*dist_0 + dist_1*dist_1 + dist_2*dist_2;


                if(dr <= squaredDist) 
                {
                    out[atomId] = 1;
                }
            }

        } // atoms indices loop


    } // sel atoms loop

}

