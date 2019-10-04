#include <cstdio>
#include <cmath>
#include <omp.h>

#include "../../libFunc.h"


void getRadialNbrDensity(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                         float *out, int outSize, float *cellDims, int nbrFrames, int sameSel, 
                         float maxR, float dr)
{
    for(int frame=0; frame < nbrFrames; ++frame)
    {
        printf("Processing frame %i of %i...    \r", frame+1, nbrFrames);

        float cD_x = cellDims[3*frame];
        float cD_y = cellDims[3*frame + 1];
        float cD_z = cellDims[3*frame + 2];

        for(int row=0; row < sel1_size; ++row)
        {
            float sel1_x = sel1[3*row*nbrFrames + 3*frame];
            float sel1_y = sel1[3*row*nbrFrames + 3*frame + 1];
            float sel1_z = sel1[3*row*nbrFrames + 3*frame + 2];

            #pragma omp parallel for
            for(int col=sameSel==1 ? row + 1 : 0; col < sel2_size; ++col)
            {
                float dist_x = sel1_x - sel2[3*col*nbrFrames + 3*frame];
                float dist_y = sel1_y - sel2[3*col*nbrFrames + 3*frame + 1];
                float dist_z = sel1_z - sel2[3*col*nbrFrames + 3*frame + 2];

                // Apply PBC conditions
                dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
                dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
                dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

                float dist = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

                int idx = (int)(dist/dr);
                if(idx >=0 && idx < outSize)
                {
                    #pragma omp atomic
                    out[(int)(dist/dr)] += 1.0 / nbrFrames;
                }

            } // sel2 (col) loop
        } // sel1 (row) loop
    } // frames loop


}
