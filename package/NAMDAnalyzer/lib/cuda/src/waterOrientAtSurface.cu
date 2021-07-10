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
void d_waterOrientAtSurface(float *waterO, int sizeO, float *watVec, float *prot, int sizeP, 
                            float *closest, float *cellDims, int nbrFrames, 
                            float minR, float maxR, int maxN, int frame)
{
    int tx  = threadIdx.x;
    int idx = blockDim.x*blockIdx.x + tx;

    int closeId = idx * 5 * maxN;

    float cD_x = cellDims[3*frame];
    float cD_y = cellDims[3*frame + 1];
    float cD_z = cellDims[3*frame + 2];

    for(int i=0; i < maxN; ++i)
        closest[closeId + 5*i + 1] = 1000;

    if(idx < sizeO)
    { 
        float wat_x = waterO[3*nbrFrames*idx + 3*frame];
        float wat_y = waterO[3*nbrFrames*idx + 3*frame + 1];
        float wat_z = waterO[3*nbrFrames*idx + 3*frame + 2];

        float watVec_x = watVec[3*nbrFrames*idx + 3*frame];
        float watVec_y = watVec[3*nbrFrames*idx + 3*frame + 1];
        float watVec_z = watVec[3*nbrFrames*idx + 3*frame + 2];

        waterO[3*nbrFrames*idx + 3*frame + 1] = 0;

        for(int pAtom=0; pAtom < sizeP; ++pAtom)
        { 
            float dist_x = prot[3*pAtom*nbrFrames + 3*frame] - wat_x;
            float dist_y = prot[3*pAtom*nbrFrames + 3*frame + 1] - wat_y;
            float dist_z = prot[3*pAtom*nbrFrames + 3*frame + 2] - wat_z;

            // Apply PBC conditions
            dist_x = dist_x - cD_x * roundf( dist_x / cD_x );
            dist_y = dist_y - cD_y * roundf( dist_y / cD_y );
            dist_z = dist_z - cD_z * roundf( dist_z / cD_z );

            float dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

            if(dist <= maxR && dist >= minR)
            {
                waterO[3*nbrFrames*idx + 3*frame + 1] = 1;

                for(int i=0; i < maxN; ++i)
                {
                    if(dist < sqrtf(closest[closeId + 5*i + 1]))
                    {
                        for(int k=i+1; k < maxN; ++k)
                        {
                            closest[closeId + 5*k] = closest[closeId + 5*(k-1)];
                            closest[closeId + 5*k + 1] = closest[closeId + 5*(k-1) + 1];
                            closest[closeId + 5*k + 2] = closest[closeId + 5*(k-1) + 2];
                            closest[closeId + 5*k + 3] = closest[closeId + 5*(k-1) + 3];
                            closest[closeId + 5*k + 4] = closest[closeId + 5*(k-1) + 4];
                        }

                        closest[closeId + 5*i] = pAtom;
                        closest[closeId + 5*i + 1] = dist*dist;
                        closest[closeId + 5*i + 2] = dist_x;
                        closest[closeId + 5*i + 3] = dist_y;
                        closest[closeId + 5*i + 4] = dist_z;

                        break;
                    } // Inner if loop
                } // closest atom loop
            }

        } // protein atoms loop


        // Computes the vector normal to surface (addition of water to found atoms vectors,
        // weighed by their squared norm)
        float normVec_x = 0;
        float normVec_y = 0;
        float normVec_z = 0;

        for(int i=0; i < maxN; ++i)
        {
            normVec_x += closest[closeId + 5*i + 2] / closest[closeId + 5*i + 1];
            normVec_y += closest[closeId + 5*i + 3] / closest[closeId + 5*i + 1];
            normVec_z += closest[closeId + 5*i + 4] / closest[closeId + 5*i + 1];
        }


        float cosAngle = watVec_x*normVec_x + watVec_y*normVec_y + watVec_z*normVec_z;
        cosAngle /= sqrtf(watVec_x*watVec_x + watVec_y*watVec_y + watVec_z*watVec_z);
        cosAngle /= sqrtf(normVec_x*normVec_x + normVec_y*normVec_y + normVec_z*normVec_z);


        waterO[3*nbrFrames*idx + 3*frame] = cosAngle;
        waterO[3*nbrFrames*idx + 3*frame + 2] = sqrtf(closest[closeId + 1]); // Keep track of closest one

    } // conditions on kernel execution

}




void cu_waterOrientAtSurface_wrapper(float *waterO, int sizeO, float *watVec, float *prot, int sizeP, 
                                     float *out, float *cellDims, int nbrFrames, 
                                     float minR, float maxR, int maxN)
{
    // Copying waterO matrix on GPU memory
    float *cu_waterO;
    size_t size = 3 * sizeO * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_waterO, size) );
    gpuErrchk( cudaMemcpy(cu_waterO, waterO, size, cudaMemcpyHostToDevice) );


    // Copying watVec matrix on GPU memory
    float *cu_watVec;
    size = 3 * sizeO * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_watVec, size) );
    gpuErrchk( cudaMemcpy(cu_watVec, watVec, size, cudaMemcpyHostToDevice) );


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


    // An array to store the maxN closest atoms indices and distances for each water molecule
    float *cu_closest;
    size = sizeO * 5 * maxN * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_closest, size) );
    gpuErrchk( cudaMemset(cu_closest, 1000, size) );


    dim3 dimBlock( BLOCK_SIZE, 1, 1 );
    dim3 dimGrid( ceil( (float)sizeO/BLOCK_SIZE), 1, 1);

    for(int frame=0; frame < nbrFrames; ++frame)
    {
        printf("Processing frame %i of %i...     \r", frame+1, nbrFrames);

        d_waterOrientAtSurface<<<dimGrid, dimBlock>>>(cu_waterO, sizeO, cu_watVec, cu_prot, sizeP, 
                                                      cu_closest, cu_cellDims, nbrFrames, 
                                                      minR, maxR, maxN, frame);

        gpuErrchk( cudaDeviceSynchronize() );
    }


    // Copying result back into host memory
    size = 3 * sizeO * nbrFrames * sizeof(float);
    gpuErrchk( cudaMemcpy(waterO, cu_waterO, size, cudaMemcpyDeviceToHost) );


    cudaFree(cu_waterO);
    cudaFree(cu_watVec);
    cudaFree(cu_prot);
    cudaFree(cu_cellDims);
    cudaFree(cu_closest);

}


