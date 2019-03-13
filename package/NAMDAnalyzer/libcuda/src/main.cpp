#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "compIntScatFunc.h"


void cu_compIntScatFunc_wrapper(const float *atomPos, int atomPos_dim0, int atomPos_dim1, 
                                const float *qVecs, int qVecs_dim0, int qVecs_dim1,
                                int timeOri, int timeIncr, int qVec, int nbrFrames, float complex correlation,
                                cudaDeviceProp devProp);


void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float complex *out, int out_dim0, int out_dim1, 
                     int binSize, int minFrames, int maxFrames, int nbrTimeOri)
{
    int devCount;
    cudaGetDeviceCount(&devCount);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    int atom_tf_idx; 
    int atom_t0_idx;
    int qVec_idx;
    float dist_0;
    float dist_1;
    float dist_2;
    float complex exponent;

    unsigned int avgFactor = atomPos_dim0 * nbrTimeOri * qVecs_dim1; 
    unsigned int nbrBins   = (maxFrames - minFrames + 1) / binSize;

    for(int qIdx=0; qIdx < qVecs_dim0; ++qIdx)
    {
        printf("Computing q value %d of %d...\r", qIdx+1, qVecs_dim0);

        for(int bin=0; bin < nbrBins; bin+=binSize)
        {
            int nbrFrames  = minFrames + bin;
            float complex correlation = 0;

            int timeIncr = ( atomPos_dim1 - nbrFrames ) / nbrTimeOri; 

            for(int timeOri=0; timeOri < nbrTimeOri; ++timeOri)
            {

                for(int qVec=0; qVec < qVecs_dim1; ++qVec)
                {

                    #pragma omp parallel for reduction(+:correlation)
                    for(int atom=0; atom < atomPos_dim0; ++atom)
                    {

                        atom_tf_idx = 3 * (atom * atomPos_dim1 + timeOri*timeIncr + nbrFrames); 
                        atom_t0_idx = 3 * (atom * atomPos_dim1 + timeOri*timeIncr);

                        qVec_idx = 3 * (qIdx * qVecs_dim1 + qVec);

                        // Computes distances for given timestep and atom
                        dist_0 = atomPos[atom_tf_idx] - atomPos[atom_t0_idx];
                        dist_1 = atomPos[atom_tf_idx+1] - atomPos[atom_t0_idx+1];
                        dist_2 = atomPos[atom_tf_idx+2] - atomPos[atom_t0_idx+2];

                        exponent = cexpf ( I * ( qVecs[qVec_idx] * dist_0 
                                                + qVecs[qVec_idx+1] * dist_1
                                                + qVecs[qVec_idx+2] * dist_2 ) );


                        correlation += exponent;

                    } // atoms loop

                } // q vectors loop 

            } // time origins loop


            // Done with average for this q-value and time step bin 
            out[ qIdx * out_dim1 + bin ] = correlation / avgFactor; 

        } // bins loop

    } // q values loop



}
