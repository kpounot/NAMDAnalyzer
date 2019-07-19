#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "compIntScatFunc.h"

void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float *out, int out_dim0, int out_dim1, 
                     int nbrTS, int nbrTimeOri)
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
     *          maxFrames   -> maximum number of frames to be used for timestep 
     *          nbrTimeOri  -> number of time origins to be averaged over */



    // To tackle MemoryError and because averaging cannot be done inside the exponential, a lot of loops
    // follow after this. The first one computes the intermediate scattering function for each scattering 
    // angle q. The next one computes the correlation for each time step for given q value. The last three
    // ones are used to compute the average over atoms, time origins and q vectors for the given q value and
    // time step.
    

    int nbrIter;
    int TSIncr  = (atomPos_dim1 / nbrTS);

    for(int qValId=0; qValId < qVecs_dim0; ++qValId)
    {
        printf("Computing q value %d of %d...\r", qValId+1, qVecs_dim0);

        for(int dt=0; dt < nbrTS; ++dt)
        {
            float sum_re = 0;
            float sum_im = 0;

            int timeIncr = (float)(atomPos_dim1 - dt*TSIncr) / nbrTimeOri; 

            for(int timeOri=0; timeOri < nbrTimeOri; ++timeOri)
            {

                for(int qVec=0; qVec < qVecs_dim1; ++qVec)
                {

                    #pragma omp parallel for reduction(+:sum_re, sum_im)
                    for(int atom=0; atom < atomPos_dim0; ++atom)
                    {

                        int atom_tf_idx = 3 * (atom * atomPos_dim1 + timeOri*timeIncr + dt*TSIncr); 
                        int atom_t0_idx = 3 * (atom * atomPos_dim1 + timeOri*timeIncr);

                        int qVec_idx = 3 * (qValId * qVecs_dim1 + qVec);

                        // Computes distances for given timestep and atom
                        float dist_0 = atomPos[atom_tf_idx] - atomPos[atom_t0_idx];
                        float dist_1 = atomPos[atom_tf_idx+1] - atomPos[atom_t0_idx+1];
                        float dist_2 = atomPos[atom_tf_idx+2] - atomPos[atom_t0_idx+2];

                        float re = cos( qVecs[qVec_idx] * dist_0 
                                        + qVecs[qVec_idx+1] * dist_1
                                        + qVecs[qVec_idx+2] * dist_2 );

                        float im = sin( qVecs[qVec_idx] * dist_0 
                                        + qVecs[qVec_idx+1] * dist_1
                                        + qVecs[qVec_idx+2] * dist_2 );



                        sum_re += re;
                        sum_im += im;

                    } // atoms loop

                } // q vectors loop 

            } // time origins loop

            // Done with average for this q-value and time step bin 
            out[ qValId * out_dim1 + 2*dt ]      = sum_re / (atomPos_dim0*nbrTimeOri*qVecs_dim1); 
            out[ qValId * out_dim1 + 2*dt + 1 ]  = sum_im / (atomPos_dim0*nbrTimeOri*qVecs_dim1); 

        } // dt loop

    } // q values loop

}

