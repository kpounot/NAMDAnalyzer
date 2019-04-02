#include <stdio.h>
#include <math.h>

#include "compIntScatFunc.h"

void cu_compIntScatFunc_wrapper(const float *atomPos, int atomPos_dim0, int atomPos_dim1, 
                                const float *qVecs, int qVecs_dim0, int qVecs_dim1,
                                int timeOri, int timeIncr, int qVec, int nbrFrames, float complex correlation)

void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float complex *out, int out_dim0, int out_dim1, 
                     int binSize, int minFrames, int maxFrames, int nbrTimeOri)
{
    /*  The function computes the intermediate neutron incoherent scattering function.
     *  Using given atom positions, minimum and maximum timesteps, and the desired number of time bins,
     *  vectors are substracted for each timesteps, then dotted with random q vectors and exponentiated.
     *  This being done with averaging over q vectors, time origins and atoms. 
     *
     *  Input:  atomPos     ->  flattened 3D array of atom positions for each frame
     *                          dimensions: atoms (0), frames (1), coordinates (2)
     *          qVecs       ->  flattened 3D array of qVectors
     *                          dimensions: q index (0), q vectors (1), coordinates (2) 
     *          out         ->  output array of floats to store correlations for each q-value (dim 0) and
     *                          each timestep bin (dim 1) 
     *          cellDims    ->  dimensions of the cell in x, y and z (used for periodic boundary conditions) 
     *          binSize     -> increment for the number of frames corresponding to time interval
     *          minFrames   -> minimum number of frames to be used for timestep
     *          maxFrames   -> maximum number of frames to be used for timestep 
     *          nbrTimeOri  -> number of time origins to be averaged over */



    // To tackle MemoryError and because averaging cannot be done inside the exponential, a lot of loops
    // follow after this. The first one computes the intermediate scattering function for each scattering 
    // angle q. The next one computes the correlation for each time step for given q value. The last three
    // ones are used to compute the average over atoms, time origins and q vectors for the given q value and
    // time step.
    

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

