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
void d_getDistances(float *sel1, int sel1_size, float *sel2, int sel2_size, float *out, 
                    float *cellDims, int frame, int nbrFrames)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * BLOCK_SIZE + tx;
    int col = blockIdx.y * BLOCK_SIZE + ty;

    float cD_x = cellDims[3*frame];
    float cD_y = cellDims[3*frame + 1];
    float cD_z = cellDims[3*frame + 2];


    if(row < sel1_size && col < sel2_size)
    {
        float dist_x = sel1[3*row*nbrFrames + 3*frame] - sel2[3*col*nbrFrames + 3*frame];
        float dist_y = sel1[3*row*nbrFrames + 3*frame + 1] - sel2[3*col*nbrFrames + 3*frame + 1];
        float dist_z = sel1[3*row*nbrFrames + 3*frame + 2] - sel2[3*col*nbrFrames + 3*frame + 2];

        // Apply PBC conditions
        dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
        dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
        dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

        float dist = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

        out[row*sel2_size+col] += dist / nbrFrames;
    }

}




void cu_getDistances_wrapper(float *sel1, int sel1_size, float *sel2, 
                             int sel2_size, float *out, float *cellDims, int nbrFrames, int sameSel)
{
    // Copying sel1 matrix on GPU memory
    float *cu_sel1;
    size_t size = 3 * sel1_size * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_sel1, size) );
    gpuErrchk( cudaMemcpy(cu_sel1, sel1, size, cudaMemcpyHostToDevice) );

    // Copying sel2 matrix on GPU memory
    float *cu_sel2;

    if(sameSel==0)
    {
        size = 3 * sel2_size * nbrFrames * sizeof(float);
        gpuErrchk( cudaMalloc(&cu_sel2, size) );
        gpuErrchk( cudaMemcpy(cu_sel2, sel2, size, cudaMemcpyHostToDevice) );
    }
    else
        cu_sel2 = cu_sel1;


    // Copying cellDims matrix on GPU memory
    float *cu_cellDims;
    size = 3 * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    float *cu_out;
    size = sel1_size * sel2_size * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );
    gpuErrchk( cudaHostRegister(out, size, cudaHostRegisterMapped) );


    dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE, 1 );
    dim3 dimGrid( ceil( (float)sel1_size/BLOCK_SIZE), ceil( (float)sel2_size/BLOCK_SIZE), 1);

    for(int frame=0; frame < nbrFrames; ++frame)
    {
        if(frame > 0)
            printf("Processing frame %i of %i...     \r", frame+1, nbrFrames);

        d_getDistances<<<dimGrid, dimBlock>>>(cu_sel1, sel1_size, cu_sel2, sel2_size, 
                                              cu_out, cu_cellDims, frame, nbrFrames);
        gpuErrchk( cudaDeviceSynchronize() );
    }


    // Copying result back into host memory
    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaHostUnregister(out) );

    cudaFree(cu_sel1);
    cudaFree(cu_sel2);
    cudaFree(cu_cellDims);
    cudaFree(cu_out);

}
