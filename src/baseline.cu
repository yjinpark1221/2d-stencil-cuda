#include <gputk.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef RADIUS
#define RADIUS 1
#endif

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0);

// [Baseline Kernel]
// No shared memory tiling, no coarsening, unrolled for RADIUS=1
__global__ void stencil_baseline(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    // Global row and column indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if within image bounds
    if (col < width && row < height) {

#if RADIUS == 1
        float result = in[row * width + col];
        if (row + 1 >= 0 && row + 1 < height) {
          result += in[(row + 1) * width + col];
        }
        if (row - 1 >= 0 && row - 1 < height) {
          result += in[(row - 1) * width + col];
        }
        if (col + 1 >= 0 && col + 1 < width) {
          result += in[row * width + (col + 1)];
        }
        if (col - 1 >= 0 && col - 1 < width) {
          result += in[row * width + (col - 1)];
        }
#else
        float result = 0.0f;
        for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
            if (row + dy >= 0 && row + dy < height) {
                result += in[(row + dy) * width + col];
            }
        }
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            if (col + dx >= 0 && col + dx < width) {
                result += in[row * width + (col + dx)];
            }
        }
        result -= in[row * width + col]; // Subtract center element counted twice
#endif
        // Write result to global memory
        out[row * width + col] = result;
    }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *host;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  int numRows;
  int numColumns;

  args = gpuTKArg_read(argc, argv);

  host = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numRows, &numColumns);
  if (host == NULL) {
      fprintf(stderr, "Error: Failed to import data. Check input file arguments.\n");
      return -1;
  }

  hostOutput = (float *)malloc((size_t)numRows * numColumns * sizeof(float));

  gpuTKCheck(cudaMalloc(&deviceInput, (size_t)numRows * numColumns * sizeof(float)));
  gpuTKCheck(cudaMalloc(&deviceOutput, (size_t)numRows * numColumns * sizeof(float)));

  gpuTKCheck(cudaMemcpy(deviceInput, host, (size_t)numRows * numColumns * sizeof(float), cudaMemcpyHostToDevice));

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numColumns + TILE_WIDTH - 1) / TILE_WIDTH,
               (numRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  stencil_baseline<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);

  gpuTKCheck(cudaGetLastError());
  gpuTKCheck(cudaDeviceSynchronize());
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  gpuTKCheck(cudaMemcpy(hostOutput, deviceOutput, numRows * numColumns * sizeof(float), cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  gpuTKCheck(cudaFree(deviceInput));
  gpuTKCheck(cudaFree(deviceOutput));
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostOutput, numRows, numColumns);
}
