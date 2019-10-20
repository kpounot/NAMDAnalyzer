#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>

#include "../../libFunc.h"


void waterOrientAtSurface(float *waterO, int sizeO, float *watVec, float *prot, int sizeP, 
                          float *out, float *cellDims, int nbrFrames, 
                          float minR, float maxR, int maxN)
{

    float *closest = (float*) malloc(5*maxN*sizeO*sizeof(float));


    for(int frame=0; frame < nbrFrames; ++frame)
    {
        printf("Processing frame %i of %i...          \r", frame+1, nbrFrames);

        float cD_x = cellDims[3*frame];
        float cD_y = cellDims[3*frame + 1];
        float cD_z = cellDims[3*frame + 2];

        for(int i=0; i < 5*maxN*sizeO; ++i)
            closest[i] = 1000;

        #pragma omp parallel for
        for(int wIdx=0; wIdx < sizeO; ++wIdx)
        { 
            int closeId = wIdx * 5 * maxN;

            float wat_x = waterO[3*nbrFrames*wIdx + 3*frame];
            float wat_y = waterO[3*nbrFrames*wIdx + 3*frame + 1];
            float wat_z = waterO[3*nbrFrames*wIdx + 3*frame + 2];

            float watVec_x = watVec[3*nbrFrames*wIdx + 3*frame];
            float watVec_y = watVec[3*nbrFrames*wIdx + 3*frame + 1];
            float watVec_z = watVec[3*nbrFrames*wIdx + 3*frame + 2];

            waterO[3*nbrFrames*wIdx + 3*frame + 1] = 0;

            for(int pAtom=0; pAtom < sizeP; ++pAtom)
            { 
                float dist_x = prot[3*pAtom*nbrFrames + 3*frame] - wat_x;
                float dist_y = prot[3*pAtom*nbrFrames + 3*frame + 1] - wat_y;
                float dist_z = prot[3*pAtom*nbrFrames + 3*frame + 2] - wat_z;

                // Apply PBC conditions
                dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
                dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
                dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

                float dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

                if(dist <= maxR && dist >= minR)
                {
                    waterO[3*nbrFrames*wIdx + 3*frame + 1] = 1;

                    for(int i=0; i < maxN; ++i)
                    {
                        if(dist < sqrtf(closest[closeId + 5*i + 1]))
                        {
                            for(int k=i+1; k < maxN; ++k)
                            {
                                closest[closeId + 5*k] = closest[closeId + 5*(k-1)];
                                closest[closeId + 5*k + 1] = closest[closeId + 5*(k-1) + 1];
                                closest[closeId + 5*k + 2] = closest[closeId + 5*(k-1) + 2];
                                closest[closeId + 5*k + 3] = closest[closeId + 5*(k-1) + 3];
                                closest[closeId + 5*k + 4] = closest[closeId + 5*(k-1) + 4];
                            }

                            closest[closeId + 5*i] = pAtom;
                            closest[closeId + 5*i + 1] = dist*dist;
                            closest[closeId + 5*i + 2] = dist_x;
                            closest[closeId + 5*i + 3] = dist_y;
                            closest[closeId + 5*i + 4] = dist_z;

                            break;
                        } // Inner if loop
                    } // closest atom loop
                }

            } // protein atoms loop

            // Computes the vector normal to surface (addition of water to found atoms vectors,
            // weighed by their squared norm)
            float normVec[3] = {0, 0, 0};

            for(int i=0; i < maxN; ++i)
            {
                normVec[0] += closest[closeId + 5*i + 2] / closest[closeId + 5*i + 1];
                normVec[1] += closest[closeId + 5*i + 3] / closest[closeId + 5*i + 1];
                normVec[2] += closest[closeId + 5*i + 4] / closest[closeId + 5*i + 1];
            }


            float cosAngle = watVec_x*normVec[0] + watVec_y*normVec[1] + watVec_z*normVec[2];
            cosAngle /= sqrtf(watVec_x*watVec_x + watVec_y*watVec_y + watVec_z*watVec_z);
            cosAngle /= sqrtf(normVec[0]*normVec[0] + normVec[1]*normVec[1] + normVec[2]*normVec[2]);


            waterO[3*nbrFrames*wIdx + 3*frame] = cosAngle;
            waterO[3*nbrFrames*wIdx + 3*frame + 2] = sqrtf(closest[closeId + 1]); // Keep track of closest one
        } // loops on waters

    } // loop on frames

    free(closest);

}

