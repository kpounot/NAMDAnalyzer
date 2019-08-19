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
void compWithin( float *allAtoms, int nbrAtoms, int nbrFrames, int *selAtoms, int sel_size,
                    int *out, float *cellDims, float distance, int frame )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int refIdx = 0;
    
    if(idx < sel_size)
    {
        refIdx = 3 * nbrFrames * selAtoms[idx] + 3 * frame;
        atomicExch( &out[ selAtoms[idx]*nbrFrames + frame], 1 );
    

        for(int atom=(nbrAtoms+powf(-1, idx)*nbrAtoms) / 2; 
                atom < nbrAtoms && atom > -1; 
                atom += powf(-1, idx+1))
        {
            if(out[nbrFrames*atom + frame] == 0)
            {
                // Computes distances for given timestep and atom
                float dist_x = allAtoms[3 * nbrFrames * atom + 3*frame] - allAtoms[refIdx];
                float dist_y = allAtoms[3 * nbrFrames * atom + 3*frame + 1] - allAtoms[refIdx + 1];
                float dist_z = allAtoms[3 * nbrFrames * atom + 3*frame + 2] - allAtoms[refIdx + 2];

                // Applying PBC corrections
                dist_x = dist_x - cellDims[3*frame] * roundf( dist_x / cellDims[3*frame] );
                dist_y = dist_y - cellDims[3*frame+1] * roundf( dist_y / cellDims[3*frame+1] );
                dist_z = dist_z - cellDims[3*frame+2] * roundf( dist_z / cellDims[3*frame+2] );


                float dist = sqrtf(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z); 

                if(dist <= distance)
                    atomicExch(&out[atom*nbrFrames + frame], 1);
            }
        }
    }

}





void cu_getWithin_wrapper(  float *allAtoms, int nbrAtoms, int nbrFrames, int *selAtoms, int sel_size,
                                float *cellDims, int *out, float distance )
{
    // Copying atom1 matrix on GPU memory
    float *cu_allAtoms;
    size_t size = 3 * nbrAtoms * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_allAtoms, size) );
    gpuErrchk( cudaMemcpy(cu_allAtoms, allAtoms, size, cudaMemcpyHostToDevice) );

    // Copying atoms2 matrix on GPU memory
    int *cu_selAtoms;
    size = sel_size * sizeof(int);
    gpuErrchk( cudaMalloc(&cu_selAtoms, size) );
    gpuErrchk( cudaMemcpy(cu_selAtoms, selAtoms, size, cudaMemcpyHostToDevice) );

    //Copying cellDims on GPU memory
    float *cu_cellDims;
    size = 3 * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    int *cu_out;
    size = nbrAtoms * nbrFrames * sizeof(int);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );


    int nbrBlocks = ceilf( (float)sel_size / BLOCK_SIZE );

    for(int frame=0; frame < nbrFrames; ++frame)
    {
        compWithin<<<nbrBlocks, BLOCK_SIZE>>>(cu_allAtoms, nbrAtoms, nbrFrames, cu_selAtoms, sel_size, 
                                                                cu_out, cu_cellDims, distance, frame);
        gpuErrchk( cudaDeviceSynchronize() );
    }


    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );


    cudaFree(cu_allAtoms);
    cudaFree(cu_selAtoms);
    cudaFree(cu_out);
    cudaFree(cu_cellDims);
}
