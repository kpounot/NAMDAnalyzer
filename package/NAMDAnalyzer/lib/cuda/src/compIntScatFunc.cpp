#include "../../libFunc.h"

void cu_compIntScatFunc_wrapper(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                                float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                                float *out, int nbrTS, int nbrTimeOri, float *scatLength);


void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float *out, int out_dim0, int out_dim1, 
                     int nbrTS, int nbrTimeOri, float *scatLength)
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


    cu_compIntScatFunc_wrapper(atomPos, atomPos_dim0, atomPos_dim1, atomPos_dim2, qVecs, qVecs_dim0, 
                        qVecs_dim1, qVecs_dim2, out, nbrTS, nbrTimeOri, scatLength);
}

