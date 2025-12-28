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
  } while (0)

// [Tiled Kernel]
// Shared memory tiling with halo regions, input tiling, no coasening, unrolled for RADIUS=1 and RADIUS=2
__global__ void stencil_tiled(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
  __shared__ float tile[TILE_WIDTH + 2 * RADIUS][TILE_WIDTH + 2 * RADIUS];
  // Calculate global row and column indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col_o = blockIdx.x * TILE_WIDTH + tx;
  int row_o = blockIdx.y * TILE_WIDTH + ty;
  int col_i = col_o - RADIUS;
  int row_i = row_o - RADIUS;
  // Load data into shared memory with halo regions
  if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
    tile[ty][tx] = in[row_i * width + col_i];
  } else {
    tile[ty][tx] = 0.0f; // Handle out-of-bounds
  }
  __syncthreads();
  // Perform stencil operation
  if (tx < TILE_WIDTH && ty < TILE_WIDTH &&
      row_o >= 0 && col_o >= 0 &&
      row_o < height && col_o < width) {

    float result = 0.0f;
#if RADIUS == 1
    result += tile[ty - 1 + RADIUS][tx + RADIUS]; // Top
    result += tile[ty + 1 + RADIUS][tx + RADIUS]; // Bottom
    result += tile[ty + RADIUS][tx - 1 + RADIUS]; // Left
    result += tile[ty + RADIUS][tx + 1 + RADIUS]; // Right
    result += tile[ty + RADIUS][tx + RADIUS];     // Center
#elif RADIUS == 2
    result += tile[ty - 2 + RADIUS][tx + RADIUS]; // Top 2
    result += tile[ty - 1 + RADIUS][tx + RADIUS]; // Top 1
    result += tile[ty + 1 + RADIUS][tx + RADIUS]; // Bottom 1
    result += tile[ty + 2 + RADIUS][tx + RADIUS]; // Bottom 2
    result += tile[ty + RADIUS][tx - 2 + RADIUS]; // Left 2
    result += tile[ty + RADIUS][tx - 1 + RADIUS]; // Left 1
    result += tile[ty + RADIUS][tx + 1 + RADIUS]; // Right 1
    result += tile[ty + RADIUS][tx + 2 + RADIUS]; // Right 2
    result += tile[ty + RADIUS][tx + RADIUS];     // Center
#else
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
      result += tile[ty + RADIUS + dy][tx + RADIUS];
    }
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
      result += tile[ty + RADIUS][tx + RADIUS + dx];
    }
    result -= tile[ty + RADIUS][tx + RADIUS]; // Subtract center element counted twice
#endif
    out[row_o * width + col_o] = result;
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

  dim3 dimGrid((numColumns + TILE_WIDTH - 1) / TILE_WIDTH,
               (numRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);
  dim3 dimBlock(TILE_WIDTH + 2 * RADIUS, TILE_WIDTH + 2 * RADIUS, 1);
  fprintf(stdout, "CUDA kernel launch with %d blocks of %d threads\n", dimGrid.x * dimGrid.y, dimBlock.x * dimBlock.y);
  fprintf(stdout, "Image size: %d x %d\n", numColumns, numRows);
  fprintf(stdout, "Tile size: %d x %d\n", TILE_WIDTH, TILE_WIDTH);
  fprintf(stdout, "Block size: %d x %d\n", dimBlock.x, dimBlock.y);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  stencil_tiled<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);

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

  free(host);
  free(hostOutput);
  return 0;
}
