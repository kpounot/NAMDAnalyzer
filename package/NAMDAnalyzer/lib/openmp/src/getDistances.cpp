#include <cstdio>
#include <cmath>
#include <list>
#include <omp.h>

#include "../../libFunc.h"


void getDistances(float *sel1, int size_sel1, float *sel2, int size_sel2, float *cellDims, float *out)
{
    /*  The function computes, for each atom in sel, the distance with all other atoms in atomPos.
     *  Then, Periodic Boundary Conditions (PBC) are applied using cell dimensions in cellDims.
     *
     *  Input:  sel1        -> first selection of atoms coordinates
     *          size_sel1   -> size of sel1 list, in number of elements
     *          sel2        -> second selection of atom coordinates, distances with sel1 will be computed
     *          size_sel2   -> size of sel2 list, in number of elements
     *          cellDims    -> dimensions of the periodic cell to apply periodic boundary conditions
     *          out         -> output matrix to store the result */



    for(int sel1_idx=0; sel1_idx < size_sel1; ++sel1_idx)
    {

        #pragma omp parallel for
        for(int sel2_idx=0; sel2_idx < size_sel2; ++sel2_idx)
        {
            // Computes distances for given timestep and atom
            float dist_x = sel2[3 * sel2_idx] - sel1[3 * sel1_idx];
            float dist_y = sel2[3 * sel2_idx + 1] - sel1[3 * sel1_idx + 1];
            float dist_z = sel2[3 * sel2_idx + 2] - sel1[3 * sel1_idx + 2];

            // Applying PBC corrections
            dist_x = dist_x - cellDims[0] * round( dist_x / cellDims[0] );
            dist_y = dist_y - cellDims[1] * round( dist_y / cellDims[1] );
            dist_z = dist_z - cellDims[2] * round( dist_z / cellDims[2] );


            out[sel1_idx * size_sel2 + sel2_idx] = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 

        } // sel2 loop


    } // sel1 loop

}

