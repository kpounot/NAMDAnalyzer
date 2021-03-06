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


/*__________________________________________________

  Hydrogen bond autocorrelation
  __________________________________________________ */

__global__
void d_getHBCorr(float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors, float *hydrogens, int size_hydrogens, 
                        float *cellDims,
                        float *out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float cosAngle, int continuous )
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;


    for(int dt=0; dt < nbrFrames; ++dt)
    {
        if( row < size_acceptors && col < size_donors )
        {
            float cD_x = cellDims[3*dt];
            float cD_y = cellDims[3*dt + 1];
            float cD_z = cellDims[3*dt + 2];

            // Computes distances for given timestep and atom
            float hyd_x = hydrogens[3 * nbrFrames * col + 3 * dt];
            float acc_x = acceptors[3 * nbrFrames * row + 3 * dt]; 
            float h_acc_x = hyd_x - acc_x;
            h_acc_x = h_acc_x - cD_x * roundf(h_acc_x / cD_x);

            float hyd_y = hydrogens[3 * nbrFrames * col + 3 * dt + 1]; 
            float acc_y = acceptors[3 * nbrFrames * row + 3 * dt + 1]; 
            float h_acc_y = hyd_y - acc_y;
            h_acc_y = h_acc_y - cD_y * roundf(h_acc_y / cD_y);

            float hyd_z = hydrogens[3 * nbrFrames * col + 3 * dt + 2]; 
            float acc_z = acceptors[3 * nbrFrames * row + 3 * dt + 2]; 
            float h_acc_z = hyd_z - acc_z;
            h_acc_z = h_acc_z - cD_z * roundf(h_acc_z / cD_z);


            float don_x = donors[3 * nbrFrames * col + 3 * dt]; 
            float don_y = donors[3 * nbrFrames * col + 3 * dt + 1]; 
            float don_z = donors[3 * nbrFrames * col + 3 * dt + 2]; 

            float acc_d_x = acc_x - don_x;
            float acc_d_y = acc_y - don_y;
            float acc_d_z = acc_z - don_z;


            float dist = sqrtf(h_acc_x*h_acc_x + h_acc_y*h_acc_y + h_acc_z*h_acc_z); 

            float angle = (h_acc_x * acc_d_x + h_acc_y * acc_d_y + h_acc_z * acc_d_z);
            angle /= ( dist * sqrtf(acc_d_x*acc_d_x + acc_d_y*acc_d_y + acc_d_z*acc_d_z) );

            if(dist <= maxR && angle <= cosAngle)
                atomicAdd( &out[dt], 1 );

            else
            {
                if(dt == 0) 
                    break;

                if(continuous==1)
                    break;
            }

        } // if loop, matrix boundaries
    } // time steps loop

}





void getHBCorr_wrapper(  float *acceptors, int size_acceptors, int nbrFrames,
                                float *donors, int size_donors,
                                float *hydrogens, int size_hydrogens, 
                                float *cellDims,
                                float *out, int maxTime, int step, 
                                int nbrTimeOri, float maxR, float minAngle, int continuous )
{

    float cosAngle = cosf(minAngle);

    // Copying acceptors matrix on GPU memory
    float *cu_acceptors;
    size_t size = 3 * size_acceptors * nbrFrames * sizeof(float);
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

    // Copying hydrogens matrix on GPU memory
    float *cu_cellDims;
    size = 3 * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    float *cu_out;
    size = nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );



    dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE, 1 );
    dim3 dimGrid( ceil( (float)size_donors/BLOCK_SIZE), ceil( (float)size_acceptors/BLOCK_SIZE), 1);

    d_getHBCorr<<<dimGrid, dimBlock>>>(cu_acceptors, size_acceptors, 
                                       nbrFrames, cu_donors, 
                                       size_donors, cu_hydrogens, size_hydrogens, 
                                       cu_cellDims,
                                       cu_out, maxTime, step, nbrTimeOri, 
                                       maxR, cosAngle, continuous);
    gpuErrchk( cudaDeviceSynchronize() );

    // Copying result back into host memory
    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );

    cudaFree(cu_acceptors);
    cudaFree(cu_donors);
    cudaFree(cu_hydrogens);
    cudaFree(cu_cellDims);
    cudaFree(cu_out);
}





/*__________________________________________________

  Hydrogen bond number
  __________________________________________________ */


__global__
void d_getHBNbr(float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors, float *hydrogens, int size_hydrogens, 
                        float *cellDims, float *out, float maxR, float cosAngle,
                        int frame )
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;


    if( row < size_acceptors && col < size_donors )
    {
        float cD_x = cellDims[3*frame];
        float cD_y = cellDims[3*frame + 1];
        float cD_z = cellDims[3*frame + 2];

        // Computes distances for given timestep and atom
        float hyd_x = hydrogens[3 * nbrFrames * col + 3 * frame];
        float acc_x = acceptors[3 * nbrFrames * row + 3 * frame]; 
        float h_acc_x = hyd_x - acc_x;
        h_acc_x = h_acc_x - cD_x * roundf(h_acc_x / cD_x);

        float hyd_y = hydrogens[3 * nbrFrames * col + 3 * frame + 1]; 
        float acc_y = acceptors[3 * nbrFrames * row + 3 * frame + 1]; 
        float h_acc_y = hyd_y - acc_y;
        h_acc_y = h_acc_y - cD_y * roundf(h_acc_y / cD_y);

        float hyd_z = hydrogens[3 * nbrFrames * col + 3 * frame + 2]; 
        float acc_z = acceptors[3 * nbrFrames * row + 3 * frame + 2]; 
        float h_acc_z = hyd_z - acc_z;
        h_acc_z = h_acc_z - cD_z * roundf(h_acc_z / cD_z);


        float don_x = donors[3 * nbrFrames * col + 3 * frame]; 
        float don_y = donors[3 * nbrFrames * col + 3 * frame + 1]; 
        float don_z = donors[3 * nbrFrames * col + 3 * frame + 2]; 

        float acc_d_x = acc_x - don_x;
        float acc_d_y = acc_y - don_y;
        float acc_d_z = acc_z - don_z;


        float dist = sqrtf(h_acc_x*h_acc_x + h_acc_y*h_acc_y + h_acc_z*h_acc_z); 

        float angle = (h_acc_x * acc_d_x + h_acc_y * acc_d_y + h_acc_z * acc_d_z);
        angle /= ( dist * sqrtf(acc_d_x*acc_d_x + acc_d_y*acc_d_y + acc_d_z*acc_d_z) );

        if(dist <= maxR && angle <= cosAngle)
            atomicAdd( &out[frame], 1 );


    } // if loop, matrix boundaries

}



void getHBNbr_wrapper(  float *acceptors, int size_acceptors, int nbrFrames,
                                float *donors, int size_donors,
                                float *hydrogens, int size_hydrogens, 
                                float *cellDims,
                                float *out, 
                                float maxR, float minAngle )
{

    float cosAngle = cosf(minAngle);

    // Copying acceptors matrix on GPU memory
    float *cu_acceptors;
    size_t size = 3 * size_acceptors * nbrFrames * sizeof(float);
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

    // Copying hydrogens matrix on GPU memory
    float *cu_cellDims;
    size = 3 * nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_cellDims, size) );
    gpuErrchk( cudaMemcpy(cu_cellDims, cellDims, size, cudaMemcpyHostToDevice) );

    // Copying out matrix on GPU memory
    float *cu_out;
    size = nbrFrames * sizeof(float);
    gpuErrchk( cudaMalloc(&cu_out, size) );
    gpuErrchk( cudaMemset(cu_out, 0, size) );


    dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE, 1 );
    dim3 dimGrid( ceil( (float)size_donors/BLOCK_SIZE), ceil( (float)size_acceptors/BLOCK_SIZE), 1);

    for(int frame=0; frame < nbrFrames; ++frame)
    {
        printf("Processing frame %i of %i...        \r", frame+1, nbrFrames);

        d_getHBNbr<<<dimGrid, dimBlock>>>(cu_acceptors, size_acceptors, 
                                           nbrFrames, cu_donors, 
                                           size_donors, cu_hydrogens, size_hydrogens, 
                                           cu_cellDims, cu_out, maxR, cosAngle, frame);
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // Copying result back into host memory
    gpuErrchk( cudaMemcpy(out, cu_out, size, cudaMemcpyDeviceToHost) );

    cudaFree(cu_acceptors);
    cudaFree(cu_donors);
    cudaFree(cu_hydrogens);
    cudaFree(cu_cellDims);
    cudaFree(cu_out);
}
