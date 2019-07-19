#include <stdio.h>

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
void compDistances( float *atoms1, int atoms1_size, float *atoms2, int atoms2_size,
                    float *out, float *cellDims )
{
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    if(row < atoms1_size && col < atoms2_size)
    {
        // Computes distances for given timestep and atom
        float dist_x = atoms2[3 * col] - atoms1[3 * row];
        float dist_y = atoms2[3 * col + 1] - atoms1[3 * row + 1];
        float dist_z = atoms2[3 * col + 2] - atoms1[3 * row + 2];

        // Applying PBC corrections
        dist_x = dist_x - cellDims[0] * round( dist_x / cellDims[0] );
        dist_y = dist_y - cellDims[1] * round( dist_y / cellDims[1] );
        dist_z = dist_z - cellDims[2] * round( dist_z / cellDims[2] );


        out[row*atoms2_size + col] = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 
    }

    __syncthreads();

}





void cu_getDistances_wrapper(  float *atoms1, int atoms1_size, float *atoms2, int atoms2_size,
                                float *cellDims, float *out )
{
    // Copying atom1 matrix on GPU memory
    float *cu_atoms1;
    size_t size1 = 3 * atoms1_size * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_atoms1, size1) );
    gpuErrchk( cudaMemcpy(cu_atoms1, atoms1, size1, cudaMemcpyHostToDevice) );

    // Copying atoms2 matrix on GPU memory
    float *cu_atoms2;
    size_t size2 = 3 * atoms2_size * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_atoms2, size2) );
    gpuErrchk( cudaMemcpy(cu_atoms2, atoms2, size2, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    float *cu_out;
    size_t size_out = atoms1_size * atoms2_size * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_out, size_out) );
    gpuErrchk( cudaMemset(cu_out, 0, size_out) );

    //Copying cellDims on GPU memory
    float *cu_cellDims;
    size_t size_cellDims = 3 * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size_cellDims) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size_cellDims, cudaMemcpyHostToDevice) );

    dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE, 1 );
    dim3 dimGrid( ceil(atoms1_size/BLOCK_SIZE) + 1, ceil(atoms2_size/BLOCK_SIZE) + 1, 1);


    compDistances<<<dimGrid, dimBlock>>>(cu_atoms1, atoms1_size, cu_atoms2, atoms2_size, cu_out, cu_cellDims);
    gpuErrchk( cudaDeviceSynchronize() );


    gpuErrchk( cudaMemcpy(out, cu_out, size_out, cudaMemcpyDeviceToHost) );


    cudaFree(cu_atoms1);
    cudaFree(cu_atoms2);
    cudaFree(cu_out);
    cudaFree(cu_cellDims);
}
