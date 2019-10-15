#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 512

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
void d_setWaterDistPBC(float *water, int sizeW, float *prot, int sizeP, 
                       float *cellDims, int nbrFrames, int nbrWAtoms, int frame)
{
    int tx  = threadIdx.x;
    int idx = blockDim.x*blockIdx.x + tx;


    float cD_x = cellDims[3*frame];
    float cD_y = cellDims[3*frame + 1];
    float cD_z = cellDims[3*frame + 2];

    float closestDist = 1000;
    float closest_x;
    float closest_y;
    float closest_z;

    float pbccorr_x;
    float pbccorr_y;
    float pbccorr_z;

    if(idx < sizeW)
    { 
        float wat_x = water[3*nbrWAtoms*nbrFrames*idx + 3*frame];
        float wat_y = water[3*nbrWAtoms*nbrFrames*idx + 3*frame + 1];
        float wat_z = water[3*nbrWAtoms*nbrFrames*idx + 3*frame + 2];

        for(int pAtom=0; pAtom < sizeP; ++pAtom)
        { 
            float prot_x = prot[3*pAtom*nbrFrames + 3*frame];
            float prot_y = prot[3*pAtom*nbrFrames + 3*frame + 1];
            float prot_z = prot[3*pAtom*nbrFrames + 3*frame + 2];

            float dist_x = wat_x - prot_x;
            float dist_y = wat_y - prot_y;
            float dist_z = wat_z - prot_z;

            // Apply PBC conditions
            pbccorr_x = cD_x * roundf( dist_x / cD_x );
            pbccorr_y = cD_y * roundf( dist_y / cD_y );
            pbccorr_z = cD_z * roundf( dist_z / cD_z );

            dist_x -= pbccorr_x;
            dist_y -= pbccorr_y;
            dist_z -= pbccorr_z;

            float dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

            if(dist < closestDist)
            {
                closestDist = dist;
                closest_x   = pbccorr_x;
                closest_y   = pbccorr_y;
                closest_z   = pbccorr_z;
            } 

        } // protein atoms loop

        // Apply PBC on position based on distance with closest
        for(int k=0; k < nbrWAtoms; ++k)
        {
            water[3*nbrWAtoms*nbrFrames*idx + 3*frame + 3*k*nbrFrames]     -= closest_x;
            water[3*nbrWAtoms*nbrFrames*idx + 3*frame + 3*k*nbrFrames + 1] -= closest_y;
            water[3*nbrWAtoms*nbrFrames*idx + 3*frame + 3*k*nbrFrames + 2] -= closest_z;
        }

    } // conditions on kernel execution

}




void cu_setWaterDistPBC_wrapper(float *water, int sizeW, float *prot, int sizeP, 
                                float *cellDims, int nbrFrames, int nbrWAtoms)
{
    // Copying waterO matrix on GPU memory
    float *cu_water;
    size_t size = 3 * nbrWAtoms * sizeW * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_water, size) );
    gpuErrchk( cudaMemcpy(cu_water, water, size, cudaMemcpyHostToDevice) );


    // Copying prot matrix on GPU memory
    float *cu_prot;
    size = 3 * sizeP * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_prot, size) );
    gpuErrchk( cudaMemcpy(cu_prot, prot, size, cudaMemcpyHostToDevice) );


    // Copying cellDims matrix on GPU memory
    float *cu_cellDims;
    size = 3 * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );


    dim3 dimBlock( BLOCK_SIZE, 1, 1 );
    dim3 dimGrid( ceil( (float)sizeW/BLOCK_SIZE), 1, 1);

    for(int frame=0; frame < nbrFrames; ++frame)
    {
        printf("Processing frame %i of %i...     \r", frame+1, nbrFrames);

        d_setWaterDistPBC<<<dimGrid, dimBlock>>>(cu_water, sizeW, cu_prot, sizeP, 
                                                 cu_cellDims, nbrFrames, nbrWAtoms, frame);

        gpuErrchk( cudaDeviceSynchronize() );
    }


    // Copying result back into host memory
    size = 3 * nbrWAtoms * sizeW * nbrFrames * sizeof(float);
    gpuErrchk( cudaMemcpy(water, cu_water, size, cudaMemcpyDeviceToHost) );


    cudaFree(cu_water);
    cudaFree(cu_prot);
    cudaFree(cu_cellDims);

}


