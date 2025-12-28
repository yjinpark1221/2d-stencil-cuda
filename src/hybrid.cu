#include <gputk.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef RADIUS
#define RADIUS 1
#endif

#ifndef COARSENING_FACTOR
#define COARSENING_FACTOR 4
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

__global__ void stencil_2D_coarsen(float* in, float* out, int width, int height) {
  const int SHARED_HEIGHT = TILE_WIDTH + 2 * RADIUS;
  const int SHARED_WIDTH = TILE_WIDTH + 2 * RADIUS;
  __shared__ float tile[SHARED_HEIGHT][SHARED_WIDTH];
  const int tile_origin_col = blockIdx.x * TILE_WIDTH - RADIUS;
  const int tile_origin_row = blockIdx.y * TILE_WIDTH - RADIUS;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  {
    const int input_col = tile_origin_col + tx;

    for (int i = 0; i < COARSENING_FACTOR; ++i) {
      const int shared_row = ty * COARSENING_FACTOR + i;
      if (shared_row >= SHARED_HEIGHT) {
        break;
      }

      const int input_row = tile_origin_row + shared_row;
      float value = 0.0f;
      if (input_row < height && input_col < width) {
        value = in[input_row * width + input_col];
      }

      tile[shared_row][tx] = value;
    }
  }

  __syncthreads();
  {
    const int output_col = tile_origin_col + tx;
    const int shared_col = tx + RADIUS;

    if (tx >= TILE_WIDTH || ty >= TILE_WIDTH / COARSENING_FACTOR || output_col >= width) {
      return;
    }

    for (int i = 0; i < COARSENING_FACTOR; ++i) {
      const int output_row = tile_origin_row + ty * COARSENING_FACTOR + i;
      const int shared_row = ty * COARSENING_FACTOR + i + RADIUS;

      if (output_row >= height) {
        continue;
      }

      float result = tile[shared_row][shared_col];
      result += tile[shared_row - 1][shared_col];
      result += tile[shared_row + 1][shared_col];
      result += tile[shared_row][shared_col + 1];
      result += tile[shared_row][shared_col - 1];
      out[output_row * width + output_col] = result;
    }
  }
}

// RADIUS=2
__global__ void stencil_2D_radius2_coarsen_unrolled(const float* __restrict__ in,
                                                    float* __restrict__ out,
                                                    int width,
                                                    int height) {
  const int SHARED_DIM = TILE_WIDTH + 2 * RADIUS;
  __shared__ float tile[SHARED_DIM][SHARED_DIM];

  const int tile_origin_col = blockIdx.x * TILE_WIDTH;
  const int tile_origin_row = blockIdx.y * TILE_WIDTH;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // =========================================================
  // Phase 1: Load Input into Shared Memory
  // =========================================================
  {
    const int input_col = tile_origin_col + tx - RADIUS;

    const int shared_row_base = ty * COARSENING_FACTOR;

    const int input_row_base = tile_origin_row + shared_row_base - RADIUS;

    // Unrolled Load (4 rows per thread)
    // --- Row 0 ---
    if (shared_row_base < SHARED_DIM) {
      float value = 0.0f;
      int r = input_row_base;
      if (input_col >= 0 && input_col < width && r >= 0 && r < height) {
        value = in[r * width + input_col];
      }
      tile[shared_row_base][tx] = value;
    }

    // --- Row 1 ---
    if (shared_row_base + 1 < SHARED_DIM) {
      float value = 0.0f;
      int r = input_row_base + 1;
      if (input_col >= 0 && input_col < width && r >= 0 && r < height) {
        value = in[r * width + input_col];
      }
      tile[shared_row_base + 1][tx] = value;
    }

    // --- Row 2 ---
    if (shared_row_base + 2 < SHARED_DIM) {
      float value = 0.0f;
      int r = input_row_base + 2;
      if (input_col >= 0 && input_col < width && r >= 0 && r < height) {
        value = in[r * width + input_col];
      }
      tile[shared_row_base + 2][tx] = value;
    }

    // --- Row 3 ---
    if (shared_row_base + 3 < SHARED_DIM) {
      float value = 0.0f;
      int r = input_row_base + 3;
      if (input_col >= 0 && input_col < width && r >= 0 && r < height) {
        value = in[r * width + input_col];
      }
      tile[shared_row_base + 3][tx] = value;
    }
  }

  __syncthreads();

  // =========================================================
  // Phase 2: Compute Output (Unrolled)
  // =========================================================
  {
    if (tx >= TILE_WIDTH || ty >= TILE_WIDTH / COARSENING_FACTOR) {
      return;
    }

    const int output_col = tile_origin_col + tx;
    if (output_col >= width) return;

    const int shared_col = tx + RADIUS;

    const int output_row_start = tile_origin_row + ty * COARSENING_FACTOR;

    const int shared_row_center_base = ty * COARSENING_FACTOR + RADIUS;

    float val_m2 = tile[shared_row_center_base - 2][shared_col]; // top-top
    float val_m1 = tile[shared_row_center_base - 1][shared_col]; // top
    float val_0  = tile[shared_row_center_base    ][shared_col]; // center
    float val_p1 = tile[shared_row_center_base + 1][shared_col]; // bottom
    float val_p2 = tile[shared_row_center_base + 2][shared_col]; // bottom-bottom

    int out_index = output_row_start * width + output_col;

    // --- Unrolled Output Row 0 ---
    {
      const int output_row = output_row_start;
      const int shared_row_center = shared_row_center_base;

      if (output_row < height) {
        float result = val_m2 + val_m1 + val_0 + val_p1 + val_p2;

        result += tile[shared_row_center][shared_col - 2];
        result += tile[shared_row_center][shared_col - 1];
        result += tile[shared_row_center][shared_col + 1];
        result += tile[shared_row_center][shared_col + 2];

        out[out_index] = result;
      }
      out_index += width;
    }

    // --- Unrolled Output Row 1 ---
    {
      // Shift Window Down
      val_m2 = val_m1;
      val_m1 = val_0;
      val_0  = val_p1;
      val_p1 = val_p2;
      val_p2 = tile[shared_row_center_base + 3][shared_col];

      const int output_row = output_row_start + 1;
      const int shared_row_center = shared_row_center_base + 1;

      if (output_row < height) {
        float result = val_m2 + val_m1 + val_0 + val_p1 + val_p2;

        result += tile[shared_row_center][shared_col - 2];
        result += tile[shared_row_center][shared_col - 1];
        result += tile[shared_row_center][shared_col + 1];
        result += tile[shared_row_center][shared_col + 2];

        out[out_index] = result;
      }
      out_index += width;
    }

    // --- Unrolled Output Row 2 ---
    {
      // Shift Window Down
      val_m2 = val_m1;
      val_m1 = val_0;
      val_0  = val_p1;
      val_p1 = val_p2;
      val_p2 = tile[shared_row_center_base + 4][shared_col];

      const int output_row = output_row_start + 2;
      const int shared_row_center = shared_row_center_base + 2;

      if (output_row < height) {
        float result = val_m2 + val_m1 + val_0 + val_p1 + val_p2;

        result += tile[shared_row_center][shared_col - 2];
        result += tile[shared_row_center][shared_col - 1];
        result += tile[shared_row_center][shared_col + 1];
        result += tile[shared_row_center][shared_col + 2];

        out[out_index] = result;
      }
      out_index += width;
    }

    // --- Unrolled Output Row 3 ---
    {
      // Shift Window Down
      val_m2 = val_m1;
      val_m1 = val_0;
      val_0  = val_p1;
      val_p1 = val_p2;
      val_p2 = tile[shared_row_center_base + 5][shared_col];

      const int output_row = output_row_start + 3;
      const int shared_row_center = shared_row_center_base + 3;

      if (output_row < height) {
        float result = val_m2 + val_m1 + val_0 + val_p1 + val_p2;

        result += tile[shared_row_center][shared_col - 2];
        result += tile[shared_row_center][shared_col - 1];
        result += tile[shared_row_center][shared_col + 1];
        result += tile[shared_row_center][shared_col + 2];

        out[out_index] = result;
      }
    }
  }
}

// RADIUS=1
__global__ void stencil_2D_coarsen_unrolled(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int width,
                                            int height) {
  const int SHARED_HEIGHT = TILE_WIDTH + 2 * RADIUS;
  const int SHARED_WIDTH = TILE_WIDTH + 2 * RADIUS;
  __shared__ float tile[SHARED_HEIGHT][SHARED_WIDTH];

  const int tile_origin_col = blockIdx.x * TILE_WIDTH;
  const int tile_origin_row = blockIdx.y * TILE_WIDTH;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // =========================================================
  // Phase 1: Load Input (Unrolled)
  // =========================================================
  {
    const int input_col = tile_origin_col + tx - RADIUS;

    const int shared_row_base = ty * COARSENING_FACTOR;

    const int input_row_base = tile_origin_row + shared_row_base - RADIUS;

    // --- Row 0 ---
    if (shared_row_base < SHARED_HEIGHT) {
      float value = 0.0f;
      int input_row = input_row_base;
      if (input_col >= 0 && input_col < width && input_row >= 0 && input_row < height) {
        value = in[input_row * width + input_col];
      }
      tile[shared_row_base][tx] = value;
    }

    // --- Row 1 ---
    if (shared_row_base + 1 < SHARED_HEIGHT) {
      float value = 0.0f;
      int input_row = input_row_base + 1;
      if (input_col >= 0 && input_col < width && input_row >= 0 && input_row < height) {
        value = in[input_row * width + input_col];
      }
      tile[shared_row_base + 1][tx] = value;
    }

    // --- Row 2 ---
    if (shared_row_base + 2 < SHARED_HEIGHT) {
      float value = 0.0f;
      int input_row = input_row_base + 2;
      if (input_col >= 0 && input_col < width && input_row >= 0 && input_row < height) {
        value = in[input_row * width + input_col];
      }
      tile[shared_row_base + 2][tx] = value;
    }

    // --- Row 3 ---
    if (shared_row_base + 3 < SHARED_HEIGHT) {
      float value = 0.0f;
      int input_row = input_row_base + 3;
      if (input_col >= 0 && input_col < width && input_row >= 0 && input_row < height) {
        value = in[input_row * width + input_col];
      }
      tile[shared_row_base + 3][tx] = value;
    }
  }

  __syncthreads();

  // =========================================================
  // Phase 2: Compute Output (Unrolled)
  // =========================================================
  {
    const int output_col = tile_origin_col + tx;

    const int shared_col = tx + RADIUS;

    if (tx >= TILE_WIDTH || ty >= TILE_WIDTH / COARSENING_FACTOR || output_col >= width) {
      return;
    }

    const int output_row_start = tile_origin_row + ty * COARSENING_FACTOR;

    const int shared_row_start = ty * COARSENING_FACTOR + RADIUS;

    float bottom = tile[shared_row_start + 1][shared_col];
    float middle = tile[shared_row_start][shared_col];
    float top = tile[shared_row_start - 1][shared_col];

    int out_index = output_row_start * width + output_col;

    // --- Row 0 ---
    {
      const int output_row = output_row_start + 0;
      const int shared_row = shared_row_start + 0;

      if (output_row < height) {
        float result = bottom + middle + top;
        result += tile[shared_row][shared_col + 1]; // Right
        result += tile[shared_row][shared_col - 1]; // Left

        out[out_index] = result;
      }
      out_index += width;
    }

    // --- Row 1 ---
    {
      const int output_row = output_row_start + 1;
      const int shared_row = shared_row_start + 1;

      if (output_row < height) {
        top = middle;
        middle = bottom;
        bottom = tile[shared_row + 1][shared_col];

        float result = bottom + middle + top;
        result += tile[shared_row][shared_col + 1]; // Right
        result += tile[shared_row][shared_col - 1]; // Left

        out[out_index] = result;
      }
      out_index += width;
    }

    // --- Row 2 ---
    {
      const int output_row = output_row_start + 2;
      const int shared_row = shared_row_start + 2;

      if (output_row < height) {
        top = middle;
        middle = bottom;
        bottom = tile[shared_row + 1][shared_col];

        float result = bottom + middle + top;
        result += tile[shared_row][shared_col + 1]; // Right
        result += tile[shared_row][shared_col - 1]; // Left

        out[out_index] = result;
      }
      out_index += width;
    }

    // --- Row 3 ---
    {
      const int output_row = output_row_start + 3;
      const int shared_row = shared_row_start + 3;

      if (output_row < height) {
        top = middle;
        middle = bottom;
        bottom = tile[shared_row + 1][shared_col];

        float result = bottom + middle + top;
        result += tile[shared_row][shared_col + 1]; // Right
        result += tile[shared_row][shared_col - 1]; // Left

        out[out_index] = result;
      }
    }
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
  dim3 dimBlock(TILE_WIDTH + 2 * RADIUS,
                (TILE_WIDTH + 2 * RADIUS + COARSENING_FACTOR - 1) / COARSENING_FACTOR,
                1);

  fprintf(stdout, "CUDA kernel launch with %d blocks of %d threads\n", dimGrid.x * dimGrid.y, dimBlock.x * dimBlock.y);
  fprintf(stdout, "Image size: %d x %d\n", numColumns, numRows);
  fprintf(stdout, "Tile size: %d x %d\n", TILE_WIDTH, TILE_WIDTH);
  fprintf(stdout, "Block size: %d x %d\n", dimBlock.x, dimBlock.y);

  gpuTKTime_start(Compute, "Performing CUDA computation");

#if RADIUS == 1
  stencil_2D_coarsen_unrolled<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);
#elif RADIUS == 2
  stencil_2D_radius2_coarsen_unrolled<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);
#else
# error "Unsupported RADIUS value"
#endif

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