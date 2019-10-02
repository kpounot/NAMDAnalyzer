#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../../libFunc.h"


void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors,
                        float *hydrogens, int size_hydrogens, 
                        float *cellDims,
                        float *out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float minAngle, int continuous)
{
    float cosAngle = cosf(minAngle);

    int row;
    int col;
    int dt;

    #pragma omp parallel for private(row, col, dt)
    for(row=0; row < size_acceptors; ++row)
    {
        for(col=row+1; col < size_donors; ++col)
        {
            int t0          = 0;
            int notBroken   = 1;

            for(dt=0; dt < nbrFrames; ++dt)
            {
                float cD_x = cellDims[3*dt];
                float cD_y = cellDims[3*dt + 1];
                float cD_z = cellDims[3*dt + 2];

                // Computes distances for given timestep and atom
                float h_acc_x = ( hydrogens[3 * nbrFrames * col + 3 * dt] 
                                - acceptors[3 * nbrFrames * row + 3 * dt] );

                float h_acc_y = ( hydrogens[3 * nbrFrames * col + 3 * dt + 1] 
                                - acceptors[3 * nbrFrames * row + 3 * dt + 1] );

                float h_acc_z = ( hydrogens[3 * nbrFrames * col + 3 * dt + 2] 
                                - acceptors[3 * nbrFrames * row + 3 * dt + 2] );

                // Applies PBC corrections
                h_acc_x = h_acc_x - cD_x * roundf( h_acc_x / cD_x );
                h_acc_y = h_acc_y - cD_y * roundf( h_acc_y / cD_y );
                h_acc_z = h_acc_z - cD_z * roundf( h_acc_z / cD_z );


                float acc_d_x = ( acceptors[3 * nbrFrames * row + 3 * dt] 
                                - donors[   3 * nbrFrames * col + 3 * dt] );
                float acc_d_y = ( acceptors[3 * nbrFrames * row + 3 * dt + 1] 
                                - donors[   3 * nbrFrames * col + 3 * dt + 1] );
                float acc_d_z = ( acceptors[3 * nbrFrames * row + 3 * dt + 2] 
                                - donors[   3 * nbrFrames * col + 3 * dt + 2] );


                float dist = sqrtf(h_acc_x*h_acc_x + h_acc_y*h_acc_y + h_acc_z*h_acc_z); 


                float angle = (h_acc_x * acc_d_x + h_acc_y * acc_d_y + h_acc_z * acc_d_z);
                angle /= ( sqrtf(h_acc_x * h_acc_x + h_acc_y * h_acc_y + h_acc_z * h_acc_z)
                            * sqrtf(acc_d_x*acc_d_x + acc_d_y*acc_d_y + acc_d_z*acc_d_z) );

                if(dist < maxR && angle < cosAngle)
                {
                    if(dt == 0)
                        t0 = 1;
                    
                    #pragma omp critical
                    out[dt] += t0 * notBroken;
                }

                else
                {
                    if(dt==0)
                        break;

                    if(continuous==1)
                        notBroken = 0;
                }

            } // time interval loop

        } // cols loop

    } // rows loop

}

