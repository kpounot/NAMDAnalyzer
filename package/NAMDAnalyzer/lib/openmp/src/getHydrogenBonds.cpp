#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../../libFunc.h"


void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors,
                        float *hydrogens, int size_hydrogens, 
                        float *cellDims, float *out, int size_out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float minAngle, int continuous)
{
    float cosAngle = cosf(minAngle);


    for(int row=0; row < size_acceptors; ++row)
    {
        for(int col=0; col < size_donors; ++col)
        {
            int t0          = 0;
            int notBroken   = 1;

            #pragma omp parallel for 
            for(int dt=0; dt < size_out; ++dt)
            {
                // Computes distances for given timestep and atom
                float h_acc_x = ( hydrogens[3 * nbrFrames * col + 3 * dt] 
                                - acceptors[3 * nbrFrames * row + 3 * dt] );
                float h_acc_y = ( hydrogens[3 * nbrFrames * col + 3 * dt + 1] 
                                - acceptors[3 * nbrFrames * row + 3 * dt + 1] );
                float h_acc_z = ( hydrogens[3 * nbrFrames * col + 3 * dt + 2] 
                                - acceptors[3 * nbrFrames * row + 3 * dt + 2] );

                float acc_d_x = ( acceptors[3 * nbrFrames * row + 3 * dt] 
                                - donors[   3 * nbrFrames * col + 3 * dt] );
                float acc_d_y = ( acceptors[3 * nbrFrames * row + 3 * dt + 1] 
                                - donors[   3 * nbrFrames * col + 3 * dt + 1] );
                float acc_d_z = ( acceptors[3 * nbrFrames * row + 3 * dt + 2] 
                                - donors[   3 * nbrFrames * col + 3 * dt + 2] );

                // Applying PBC corrections
                float dist_x = h_acc_x - cellDims[3*dt] * roundf( h_acc_x / cellDims[3*dt] );
                float dist_y = h_acc_y - cellDims[3*dt+1] * roundf( h_acc_y / cellDims[3*dt+1] );
                float dist_z = h_acc_z - cellDims[3*dt+2] * roundf( h_acc_z / cellDims[3*dt+2] );

                float dist = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 


                float angle = (h_acc_x * acc_d_x + h_acc_y * acc_d_y + h_acc_z * acc_d_z);
                angle /= ( sqrtf(h_acc_x * h_acc_x + h_acc_y * h_acc_y + h_acc_z * h_acc_z)
                            * sqrtf(acc_d_x*acc_d_x + acc_d_y*acc_d_y + acc_d_z*acc_d_z) );

                if(dist < maxR && angle < cosAngle)
                {
                    if(dt == 0)
                        t0 = 1;

                    out[dt] += t0 * notBroken;
                }

                else
                {
                    if(continuous==1)
                        notBroken = 0;
                }

            } // time interval loop

        } // cols loop

    } // rows loop

}

