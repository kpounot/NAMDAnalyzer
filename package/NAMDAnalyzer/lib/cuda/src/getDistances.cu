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
void d_getDistances(float *maxSel, int maxSize, float *minSel, int minSize, float *out, float *cellDims)
{
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    float cD_x = cellDims[0];
    float cD_y = cellDims[1];
    float cD_z = cellDims[2];

    if(row < maxSize && col < minSize)
    {
        float dist_x = maxSel[3 * row] - minSel[3*col];
        float dist_y = maxSel[3 * row + 1] - minSel[3*col + 1];
        float dist_z = maxSel[3 * row + 2] - minSel[3*col + 2];

        // Apply PBC conditions
        dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
        dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
        dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

        float dist = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

        out[row*minSize+col] = dist;
    }

}


__global__
void d_getDistances_same(float *maxSel, int maxSize, float *minSel, 
                         int minSize, float *out, float *cellDims)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;


    int n = minSize;
    int row = 0;
    while( idx - n >= 0)
    {
        row += 1;
        n   += minSize - row;
    }

    int col = minSize - 1 - ( n - idx );

    float cD_x = cellDims[0];
    float cD_y = cellDims[1];
    float cD_z = cellDims[2];

    if(row < maxSize && col < minSize)
    {
        float dist_x = maxSel[3 * row] - minSel[3*col];
        float dist_y = maxSel[3 * row + 1] - minSel[3*col + 1];
        float dist_z = maxSel[3 * row + 2] - minSel[3*col + 2];

        // Apply PBC conditions
        dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
        dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
        dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

        float dist = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

        out[row*minSize+col] = dist;
    }

}




void cu_getDistances_wrapper(float *maxSel, int maxSize, float *minSel, 
                             int minSize, float *out, float *cellDims, int sameSel)
{
    // Copying maxSel matrix on GPU memory
    float *cu_maxSel;
    size_t size = 3 * maxSize * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_maxSel, size) );
    gpuErrchk( cudaMemcpy(cu_maxSel, maxSel, size, cudaMemcpyHostToDevice) );

    // Copying minSel matrix on GPU memory
    float *cu_minSel;
    size = 3 * minSize * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_minSel, size) );
    gpuErrchk( cudaMemcpy(cu_minSel, minSel, size, cudaMemcpyHostToDevice) );

    // Copying cellDims matrix on GPU memory
    float *cu_cellDims;
    size = 3 * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    float *cu_out;
    size = maxSize * minSize * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );


    if(sameSel == 0)
    {
        dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE, 1 );
        dim3 dimGrid( ceil( (float)maxSize/BLOCK_SIZE), ceil( (float)minSize/BLOCK_SIZE), 1);

        d_getDistances<<<dimGrid, dimBlock>>>(cu_maxSel, maxSize, cu_minSel, minSize, cu_out, cu_cellDims);
        gpuErrchk( cudaDeviceSynchronize() );
    }
    else
    {
        int nbrBlocks = ceilf( (maxSize * ((float)maxSize - 1) / 2) / 512 );

        d_getDistances_same<<<nbrBlocks, 512>>>(cu_maxSel, maxSize, cu_minSel, 
                                                   minSize, cu_out, cu_cellDims);
        gpuErrchk( cudaDeviceSynchronize() );
    }


    // Copying result back into host memory
    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );

    cudaFree(cu_maxSel);
    cudaFree(cu_minSel);
    cudaFree(cu_out);
}
