#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 384

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
void compWithin( float *allAtoms, int nbrAtoms, int nbrFrames, int *selAtoms, int sel_size,
                 int *out, float *cellDims, float distance, int frame )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    float squaredDist = distance * distance;

    for(int i=idx; i < sel_size; i+=BLOCK_SIZE)
        atomicExch( &out[ selAtoms[i]*nbrFrames + frame ], 1 );

    __syncthreads();

    float cD_x  = cellDims[3*frame];
    float cD_y  = cellDims[3*frame+1];
    float cD_z  = cellDims[3*frame+2];

    if(idx < sel_size) 
    {
        int selIdx = selAtoms[idx];

        float sel_x = allAtoms[3*nbrFrames*selIdx + 3*frame];
        float sel_y = allAtoms[3*nbrFrames*selIdx + 3*frame + 1];
        float sel_z = allAtoms[3*nbrFrames*selIdx + 3*frame + 2];

        for(int atom=0; atom < nbrAtoms; ++atom)
        {
            if(out[atom*nbrFrames + frame] == 0)
            {
                float atom_x = allAtoms[3*nbrFrames*atom + 3*frame];
                float dist_x = atom_x - sel_x;
                dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
                if(dist_x > distance)
                    continue;

                float atom_y = allAtoms[3*nbrFrames*atom + 3*frame + 1];
                float dist_y = atom_y - sel_y;
                dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
                if(dist_y > distance)
                    continue;

                float atom_z = allAtoms[3*nbrFrames*atom + 3*frame + 2];
                float dist_z = atom_z - sel_z;
                dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

                float dist = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z; 

                if(dist <= squaredDist)
                {
                    atomicExch( &out[atom*nbrFrames + frame], 1 );
                }
            }
        } // atom loop
    } // thread execution condition

}





void cu_getWithin_wrapper(  float *allAtoms, int nbrAtoms, int nbrFrames, int *selAtoms, int sel_size,
                            int *out, float *cellDims, float distance )
{
    // Copying atom1 matrix on GPU memory
    float *cu_allAtoms;
    size_t size = 3 * nbrAtoms * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_allAtoms, size) );
    gpuErrchk( cudaMemcpy(cu_allAtoms, allAtoms, size, cudaMemcpyHostToDevice) );

    // Copying selAtoms array on GPU memory
    int *cu_selAtoms;
    size = sel_size * sizeof(int);
    gpuErrchk( cudaMalloc(&cu_selAtoms, size) );
    gpuErrchk( cudaMemcpy(cu_selAtoms, selAtoms, size, cudaMemcpyHostToDevice) );

    // Copying cellDims matrix on GPU memory
    float *cu_cellDims;
    size = 3 * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    int *cu_out;
    size = nbrAtoms * nbrFrames * sizeof(int);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );


    int nbrBlocks = ceilf((float)sel_size / BLOCK_SIZE);

    for(int frame=0; frame < nbrFrames; ++frame)
    {
        compWithin<<<nbrBlocks, BLOCK_SIZE>>>(cu_allAtoms, nbrAtoms, nbrFrames, 
                                              cu_selAtoms, sel_size,
                                              cu_out, cu_cellDims, distance, frame);
        gpuErrchk( cudaDeviceSynchronize() );
    }


    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );


    cudaFree(cu_allAtoms);
    cudaFree(cu_selAtoms);
    cudaFree(cu_cellDims);
    cudaFree(cu_out);
}
