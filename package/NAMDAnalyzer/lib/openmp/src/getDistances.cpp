#include <cstdio>
#include <cmath>
#include <omp.h>

#include "../../libFunc.h"



void getDistances(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                  float *out, float *cellDims, int nbrFrames, int sameSel)
{
    /*  The function computes, for each atom in sel, the distance with all other atoms in atomPos.
     *  Then, Periodic Boundary Conditions (PBC) are applied using cell dimensions in cellDims.
     *
     *  Input:  sel1      -> first selection of atoms coordinates (biggest dimension)
     *          sel1_size     -> size of sel1 list, in number of elements
     *          sel2      -> second selection of atom coordinates, distances with sel1 will be computed
     *          sel2_size     -> size of sel2 list, in number of elements
     *          out         -> output matrix to store the result */


    int max_idx;

    float dist_x, dist_y, dist_z, dist;

    for(int frame=0; frame < nbrFrames; ++frame)
    {
        if(nbrFrames > 1)
            printf("Processing frame %i of %i...    \r", frame+1, nbrFrames);

        float cD_x = cellDims[3*frame];
        float cD_y = cellDims[3*frame + 1];
        float cD_z = cellDims[3*frame + 2];

        for(max_idx=0; max_idx < sel1_size; ++max_idx)
        {
            float sel1_x = sel1[3*max_idx*nbrFrames + 3*frame];
            float sel1_y = sel1[3*max_idx*nbrFrames + 3*frame + 1];
            float sel1_z = sel1[3*max_idx*nbrFrames + 3*frame + 2];

            #pragma omp parallel for \
            shared(cD_x, cD_y, cD_z, sel1_x, sel1_y, sel1_z, max_idx) \
            private(dist_x, dist_y, dist_z, dist)
            for(int min_idx=sameSel==1 ? max_idx+1 : 0; min_idx < sel2_size; ++min_idx)
            {
                // Computes distances for given timestep and atom
                dist_x = sel2[3*min_idx*nbrFrames + 3*frame] - sel1_x;
                dist_y = sel2[3*min_idx*nbrFrames + 3*frame + 1] - sel1_y;
                dist_z = sel2[3*min_idx*nbrFrames + 3*frame + 2] - sel1_z;

                // Apply PBC conditions
                dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
                dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
                dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

                dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 
                out[max_idx * sel2_size + min_idx] = dist / nbrFrames;

            } // sel2 loop
        } // sel1 loop
    } // frames loop

}
