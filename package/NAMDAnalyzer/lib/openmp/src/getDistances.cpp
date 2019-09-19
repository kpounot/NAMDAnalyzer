#include <cstdio>
#include <cmath>
#include <list>
#include <omp.h>

#include "../../libFunc.h"


void compDistances(float *maxSel, int maxSize, float *minSel, int minSize, 
                  float *out, float *cellDims)
{
    int max_idx, min_idx;

    float cD_x = cellDims[0];
    float cD_y = cellDims[1];
    float cD_z = cellDims[2];

    for(max_idx=0; max_idx < maxSize; ++max_idx)
    {
        #pragma omp parallel for 
        for(min_idx=0; min_idx < minSize; ++min_idx)
        {
            // Computes distances for given timestep and atom
            float dist_x = minSel[3 * min_idx] - maxSel[3 * max_idx];
            float dist_y = minSel[3 * min_idx + 1] - maxSel[3 * max_idx + 1];
            float dist_z = minSel[3 * min_idx + 2] - maxSel[3 * max_idx + 2];

            // Apply PBC conditions
            dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
            dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
            dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

            out[max_idx * minSize + min_idx] = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 

        } // minSel loop
    } // maxSel loop
}


void compDistances_same(float *maxSel, int maxSize, float *minSel, int minSize, 
                        float *out, float *cellDims)
{
    float cD_x = cellDims[0];
    float cD_y = cellDims[1];
    float cD_z = cellDims[2];

    int min_idx;

    for(int max_idx=0; max_idx < maxSize; ++max_idx)
    {
        #pragma omp parallel for 
        for(min_idx=max_idx+1; min_idx < minSize; ++min_idx)
        {
            // Computes distances for given timestep and atom
            float dist_x = minSel[3 * min_idx] - maxSel[3 * max_idx];
            float dist_y = minSel[3 * min_idx + 1] - maxSel[3 * max_idx + 1];
            float dist_z = minSel[3 * min_idx + 2] - maxSel[3 * max_idx + 2];

            // Apply PBC conditions
            dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
            dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
            dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

            out[max_idx * minSize + min_idx] = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 

        } // minSel loop
    } // maxSel loop
}



void getDistances(float *maxSel, int maxSize, float *minSel, int minSize, 
                  float *out, float *cellDims, int sameSel)
{
    /*  The function computes, for each atom in sel, the distance with all other atoms in atomPos.
     *  Then, Periodic Boundary Conditions (PBC) are applied using cell dimensions in cellDims.
     *
     *  Input:  maxSel      -> first selection of atoms coordinates (biggest dimension)
     *          maxSize     -> size of maxSel list, in number of elements
     *          minSel      -> second selection of atom coordinates, distances with sel1 will be computed
     *          minSize     -> size of minSel list, in number of elements
     *          out         -> output matrix to store the result */


    if(sameSel == 0)
        compDistances(maxSel, maxSize, minSel, minSize, out, cellDims);
    else
        compDistances_same(maxSel, maxSize, minSel, minSize, out, cellDims);

}
