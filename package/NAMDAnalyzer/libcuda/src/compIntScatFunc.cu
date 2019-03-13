#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32


__global__

void compIntScatFuncKernel(const float *atomPos, int atomPos_dim0, int atomPos_dim1, 
                                const float *qVecs, int qVecs_dim0, int qVecs_dim1,
                                int timeOri, int timeIncr, int qVec, int nbrFrames, float complex correlation,
                                cudaDeviceProp devProp);
{
    int blockRow    = blockIdx.y;
    int blockCol    = blockIdx.x;
    int tx          = threadIdx.x;
    int ty          = threadIdx.y;
    int row = blockRow * BLOCK_SIZE + ty;
    int col = blockCol * BLOCK_SIZE + tx;

    float out = 0;

    for(int m=0; m < (A.width + BLOCK_SIZE - 1)/BLOCK_SIZE; ++m)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        if(row < C.height && col < C.width)
        {
            As[ty][tx] = A.elements[row * A.width + m * BLOCK_SIZE + tx];
            Bs[ty][tx] = B.elements[(m*BLOCK_SIZE + ty)*B.width + col];
        }
        else
        {
            As[ty][tx] = 0;
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        for(int i=0; i < BLOCK_SIZE; ++i)
            out += As[ty][i] * Bs[i][tx];

        __syncthreads();
    }

    if(row < C.height && col < C.width)
        C.elements[row * C.width + col] = out;
}





void cu_compIntScatFunc_wrapper(const float *atomPos, int atomPos_dim0, int atomPos_dim1, 
                                const float *qVecs, int qVecs_dim0, int qVecs_dim1,
                                int timeOri, int timeIncr, int qVec, int nbrFrames, float complex correlation,
                                cudaDeviceProp devProp);
{
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid(  (C.width + block.x - 1) / block.x,
                (C.height + block.y - 1) / block.y,
                1);

    Matrix d_A {A.width, A.height, A.stride};
    size_t size = A.height * A.width * sizeof(float);
    cudaMallocManaged(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B {B.width, B.height, B.stride};
    size = B.height * B.width * sizeof(float);
    cudaMallocManaged(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix d_C {C.width, C.height, C.stride};
    size = C.height * C.width * sizeof(float);
    cudaMallocManaged(&d_C.elements, size);
    cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);

    matMulKernel<<<grid, block>>>(d_C, d_A, d_B);

    cudaDeviceSynchronize();

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
