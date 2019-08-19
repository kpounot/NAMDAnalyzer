#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

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
void d_getHydrogenBonds(float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors, float *hydrogens, int size_hydrogens, 
                        float *cellDims, float *corr, int size_corr, int maxTime, int step, int nbrTimeOri,
                        float maxR, float cosAngle, int continuous )
{
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    int notBroken   = 1;
    int t0          = 0;

    for(int dt=0; dt < size_corr; ++dt)
    {
        if( row < size_acceptors && col < size_donors )
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

            if(dist <= maxR && angle <= cosAngle)
            {
                if(dt == 0) 
                    t0 = 1;

                atomicAdd( &corr[dt], t0 * notBroken );
            }

            else
            {
                if(continuous==1)
                    notBroken = 0;
            }

        } // if loop, matrix boundaries

    } // time interval loop

}





void getHydrogenBonds_wrapper(  float *acceptors, int size_acceptors, int nbrFrames,
                                float *donors, int size_donors,
                                float *hydrogens, int size_hydrogens, 
                                float *cellDims, float *out, int size_out, int maxTime, int step, 
                                int nbrTimeOri, float maxR, float minAngle, int continuous )
{

    float cosAngle = cosf(minAngle);

    //Copying cellDims on GPU memory
    float *cu_cellDims;
    size_t size = 3 * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying acceptors matrix on GPU memory
    float *cu_acceptors;
    size = 3 * size_acceptors * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_acceptors, size) );
    gpuErrchk( cudaMemcpy(cu_acceptors, acceptors, size, cudaMemcpyHostToDevice) );

    // Copying donors matrix on GPU memory
    float *cu_donors;
    size = 3 * size_donors * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_donors, size) );
    gpuErrchk( cudaMemcpy(cu_donors, donors, size, cudaMemcpyHostToDevice) );

    // Copying hydrogens matrix on GPU memory
    float *cu_hydrogens;
    size = 3 * size_hydrogens * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_hydrogens, size) );
    gpuErrchk( cudaMemcpy(cu_hydrogens, hydrogens, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    float *cu_out;
    size = size_out * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );

    dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE, 1 );
    dim3 dimGrid( ceil( (float)size_donors/BLOCK_SIZE), ceil( (float)size_acceptors/BLOCK_SIZE), 1);


    d_getHydrogenBonds<<<dimGrid, dimBlock>>>(cu_acceptors, size_acceptors, nbrFrames, cu_donors, 
                                         size_donors, cu_hydrogens, size_hydrogens, cu_cellDims, cu_out,
                                         size_out, maxTime, step, nbrTimeOri, maxR, cosAngle, continuous);
    gpuErrchk( cudaDeviceSynchronize() );



    // Copying result back into host memory
    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );


    cudaFree(cu_acceptors);
    cudaFree(cu_donors);
    cudaFree(cu_hydrogens);
    cudaFree(cu_out);
    cudaFree(cu_cellDims);
}
