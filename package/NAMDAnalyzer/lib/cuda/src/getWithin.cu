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
void compWithin( float *allAtoms, int nbrAtoms, float *selAtoms, int sel_size,
                    int *out, float *cellDims, float distance )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float s_sel[3*BLOCK_SIZE];
    if(idx < sel_size)
    {
        s_sel[3*tx] = selAtoms[3*idx];
        s_sel[3*tx + 1] = selAtoms[3*idx + 1];
        s_sel[3*tx + 2] = selAtoms[3*idx + 2];
    }

    __syncthreads();


    for(int atom=(nbrAtoms+powf(-1, idx)*nbrAtoms) / 2; atom < nbrAtoms && atom > -1; atom += powf(-1, idx+1))
    {
        if(out[atom] == 0 && idx < sel_size)
        {
            // Computes distances for given timestep and atom
            float dist_x = allAtoms[3 * atom] - s_sel[3 * tx];
            float dist_y = allAtoms[3 * atom + 1] - s_sel[3 * tx + 1];
            float dist_z = allAtoms[3 * atom + 2] - s_sel[3 * tx + 2];

            // Applying PBC corrections
            dist_x = dist_x - cellDims[0] * round( dist_x / cellDims[0] );
            dist_y = dist_y - cellDims[1] * round( dist_y / cellDims[1] );
            dist_z = dist_z - cellDims[2] * round( dist_z / cellDims[2] );


            float dist = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 

            if(dist <= distance)
                atomicExch(&out[atom], 1);
        }
    }

    __syncthreads();

}





void cu_getWithin_wrapper(  float *allAtoms, int nbrAtoms, float *selAtoms, int sel_size,
                                float *cellDims, int *out, float distance )
{
    // Copying atom1 matrix on GPU memory
    float *cu_allAtoms;
    size_t size = 3 * nbrAtoms * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_allAtoms, size) );
    gpuErrchk( cudaMemcpy(cu_allAtoms, allAtoms, size, cudaMemcpyHostToDevice) );

    // Copying atoms2 matrix on GPU memory
    float *cu_selAtoms;
    size = 3 * sel_size * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_selAtoms, size) );
    gpuErrchk( cudaMemcpy(cu_selAtoms, selAtoms, size, cudaMemcpyHostToDevice) );

    //Copying cellDims on GPU memory
    float *cu_cellDims;
    size = 3 * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    int *cu_out;
    size = nbrAtoms * sizeof(int);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );


    int nbrBlocks = ceil( sel_size / BLOCK_SIZE );

    compWithin<<<nbrBlocks, BLOCK_SIZE>>>(cu_allAtoms, nbrAtoms, cu_selAtoms, sel_size, 
                                                                    cu_out, cu_cellDims, distance);
    gpuErrchk( cudaDeviceSynchronize() );


    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );


    cudaFree(cu_allAtoms);
    cudaFree(cu_selAtoms);
    cudaFree(cu_out);
    cudaFree(cu_cellDims);
}
