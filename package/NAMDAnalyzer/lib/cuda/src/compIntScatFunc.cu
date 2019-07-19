#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, float *out,
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, 
                     int nbrTS, int nbrTimeOri, int qValId) 
{
    int atomId  = blockDim.x * blockIdx.x + threadIdx.x;
    int TSIncr  = (atomPos_dim1 / nbrTS);

    extern __shared__ float s_qVecs[];

    if( atomId < atomPos_dim0 )
    {

        for(int i=0; i < 3*qVecs_dim0*qVecs_dim1; ++i)
            s_qVecs[i] = qVecs[i];

    }

    
    __syncthreads();


    if( atomId < atomPos_dim0 )
    {
        for(int dt=0; dt < nbrTS; ++dt)
        {
            int timeIncr = (float)(atomPos_dim1 - dt*TSIncr) / nbrTimeOri; 

            float sum_re = 0;
            float sum_im = 0;

            for(int t0=0; t0 < nbrTimeOri; ++t0)
            {
                // Gets indices
                int atom_tf_idx = 3 * (atomId*atomPos_dim1 + t0*timeIncr + dt*TSIncr); 
                int atom_t0_idx = 3 * (atomId*atomPos_dim1 + t0*timeIncr);

                // Computes distances for given timestep and atom
                float dist_0 = atomPos[atom_tf_idx] - atomPos[atom_t0_idx];
                float dist_1 = atomPos[atom_tf_idx+1] - atomPos[atom_t0_idx+1];
                float dist_2 = atomPos[atom_tf_idx+2] - atomPos[atom_t0_idx+2];

                for(int qVecId=0; qVecId < qVecs_dim1; ++qVecId)
                {

                    int qVec_idx = 3 * (qValId * qVecs_dim1 + qVecId);

                    float re = cos( s_qVecs[qVec_idx] * dist_0 
                                    + s_qVecs[qVec_idx+1] * dist_1
                                    + s_qVecs[qVec_idx+2] * dist_2 );

                    float im = sin( s_qVecs[qVec_idx] * dist_0 
                                    + s_qVecs[qVec_idx+1] * dist_1
                                    + s_qVecs[qVec_idx+2] * dist_2 );

                    sum_re += re;
                    sum_im += im;

                } // q vectors loop
            } // time origins loop 

            atomicAdd( &(out[2*(qValId*nbrTS + dt)]), sum_re / (nbrTS*qVecs_dim1) );
            atomicAdd( &(out[2*(qValId*nbrTS + dt) + 1]), sum_im / (nbrTS*qVecs_dim1) );

        } // time increments loop

    } // condition on atom index

}





void cu_compIntScatFunc_wrapper(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                                float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                                float *out, int nbrTS, int nbrTimeOri)
{
    // Copying atomPos matrix on GPU memory
    float *cu_atomPos;
    size_t size_atomPos = atomPos_dim0 * atomPos_dim1 * atomPos_dim2 * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_atomPos, size_atomPos) );
    gpuErrchk( cudaMemcpy(cu_atomPos, atomPos, size_atomPos, cudaMemcpyHostToDevice) );

    // Copying qVecs matrix on GPU memory
    float *cu_qVecs;
    size_t size_qVecs = qVecs_dim0 * qVecs_dim1 * qVecs_dim2 * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_qVecs, size_qVecs) );
    gpuErrchk( cudaMemcpy(cu_qVecs, qVecs, size_qVecs, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    float *cu_out;
    size_t size_out = 2 * qVecs_dim0 * nbrTS * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_out, size_out) );
    gpuErrchk( cudaMemset(cu_out, 0, size_out) );

    int nbrBlocks = ceil((float)atomPos_dim0 / BLOCK_SIZE);
    int sharedMemSize = sizeof(float) * 3 * qVecs_dim0 * qVecs_dim1; 


    // Starts computation of intermediate scattering function
    for(int qValId=0; qValId < qVecs_dim0; ++qValId)
    {
        compIntScatFunc<<<nbrBlocks, BLOCK_SIZE, sharedMemSize>>>(cu_atomPos, atomPos_dim0, 
                                                            atomPos_dim1, cu_out, cu_qVecs, 
                                                            qVecs_dim0, qVecs_dim1, nbrTS, 
                                                            nbrTimeOri, qValId);
        gpuErrchk( cudaDeviceSynchronize() );
    }



    cudaMemcpy(out, cu_out, size_out, cudaMemcpyDeviceToHost);

    for(int i=0; i < 2*nbrTS*qVecs_dim0; ++i)
    {
        out[i] /= atomPos_dim0;
    }

    cudaFree(cu_atomPos);
    cudaFree(cu_qVecs);
    cudaFree(cu_out);
}
