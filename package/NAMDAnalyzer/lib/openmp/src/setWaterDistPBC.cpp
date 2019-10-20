#include <stdio.h>
#include <math.h>

#include <omp.h>

#include "../../libFunc.h"


void setWaterDistPBC(float *water, int sizeW, float *prot, int sizeP, 
                     float *cellDims, int nbrFrames, int nbrWAtoms)
{

    float closestDist = 1000;
    float closest_x   = 0;
    float closest_y   = 0;
    float closest_z   = 0;

    float pbccorr_x;
    float pbccorr_y;
    float pbccorr_z;

    for(int frame=0; frame < nbrFrames; ++frame)
    { 
        printf("Processing frame %i of %i...          \r", frame+1, nbrFrames);

        float cD_x = cellDims[3*frame];
        float cD_y = cellDims[3*frame + 1];
        float cD_z = cellDims[3*frame + 2];

        for(int wIdx=0; wIdx < sizeW; ++wIdx)
        {
            float wat_x = water[3*nbrWAtoms*nbrFrames*wIdx + 3*frame];
            float wat_y = water[3*nbrWAtoms*nbrFrames*wIdx + 3*frame + 1];
            float wat_z = water[3*nbrWAtoms*nbrFrames*wIdx + 3*frame + 2];

            for(int pAtom=0; pAtom < sizeP; ++pAtom)
            { 
                float prot_x = prot[3*pAtom*nbrFrames + 3*frame];
                float prot_y = prot[3*pAtom*nbrFrames + 3*frame + 1];
                float prot_z = prot[3*pAtom*nbrFrames + 3*frame + 2];

                float dist_x = wat_x - prot_x;
                float dist_y = wat_y - prot_y;
                float dist_z = wat_z - prot_z;

                // Apply PBC conditions
                pbccorr_x = cD_x * roundf( dist_x / cD_x );
                pbccorr_y = cD_y * roundf( dist_y / cD_y );
                pbccorr_z = cD_z * roundf( dist_z / cD_z );

                dist_x -= pbccorr_x;
                dist_y -= pbccorr_y;
                dist_z -= pbccorr_z;

                float dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

                if(dist < closestDist)
                {
                    closestDist = dist;
                    closest_x   = pbccorr_x;
                    closest_y   = pbccorr_y;
                    closest_z   = pbccorr_z;
                } 

            } // protein atoms loop

            // Apply PBC on position based on distance with closest
            for(int k=0; k < nbrWAtoms; ++k)
            {
                water[3*nbrWAtoms*nbrFrames*wIdx + 3*frame + 3*k*nbrFrames]     -= closest_x;
                water[3*nbrWAtoms*nbrFrames*wIdx + 3*frame + 3*k*nbrFrames + 1] -= closest_y;
                water[3*nbrWAtoms*nbrFrames*wIdx + 3*frame + 3*k*nbrFrames + 2] -= closest_z;
            }
        }

    } // loop on frames
}


