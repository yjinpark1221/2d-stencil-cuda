#include <gputk.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// COARSENING FACTOR : 4 (hard-coded loop unrolling)

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


// [Thread Coarsened with Global Memory]
__global__ void stencil_2D_global_coarsen_2(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
  // Global row and column indices
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row_base = (blockIdx.y * blockDim.y + threadIdx.y) * 2; // Coarsening by factor of 2 in row direction

  // Process 2 pixels in the row direction
  int row = row_base;
  float result = 0.0f;
  float bottom = 0.0f, middle = 0.0f, top = 0.0f;

  if (col < width) {
    // 1st pixel (row, col)
    if (row - 1 >= 0 && row - 1 < height) {
      top = in[(row - 1) * width + col];
    }
    if (row >= 0 && row < height) {
      middle = in[row * width + col];
    }
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }

    if (row < height) {
      result = top + middle + bottom;

      if (col - 1 >= 0) {
        result += in[row * width + (col - 1)];
      }
      if (col + 1 < width) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 2nd pixel (row + 1, col)
    row = row_base + 1;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }

    if (row < height) {
      result = top + middle + bottom;

      if (col - 1 >= 0) {
        result += in[row * width + (col - 1)];
      }
      if (col + 1 < width) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }
  }
}

__global__ void stencil_2D_global_coarsen_4_radius_2(const float* __restrict__ in, float* __restrict__ out, const int width, const int height) {
  // 1. Thread & Global Index Calculation
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_base = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

  // 2. Horizontal Boundary Checks (Radius 2)
  const bool has_L2 = (col - 2) >= 0;
  const bool has_L1 = (col - 1) >= 0;
  const bool has_R1 = (col + 1) < width;
  const bool has_R2 = (col + 2) < width;

  // 3. Register Window Initialization
  // 수직 방향 처리를 위한 레지스터 윈도우 (row-2 ~ row+2)
  // 초기 로딩: 첫 번째 픽셀(row_base) 처리를 위해 미리 주변부를 읽어옴
  float val_m2 = 0.0f; // row - 2
  float val_m1 = 0.0f; // row - 1
  float val_0  = 0.0f; // row (center)
  float val_p1 = 0.0f; // row + 1
  float val_p2 = 0.0f; // row + 2

  if (col < width) {
    // ---- [Prolog: Load initial window for the 1st pixel] ----
    // Base row is 'row_base'
    if (row_base - 2 >= 0 && row_base - 2 < height) val_m2 = in[(row_base - 2) * width + col];
    if (row_base - 1 >= 0 && row_base - 1 < height) val_m1 = in[(row_base - 1) * width + col];
    if (row_base     >= 0 && row_base     < height) val_0  = in[(row_base    ) * width + col];
    if (row_base + 1 >= 0 && row_base + 1 < height) val_p1 = in[(row_base + 1) * width + col];
    if (row_base + 2 >= 0 && row_base + 2 < height) val_p2 = in[(row_base + 2) * width + col];

    float result;
    int current_row;

    // =========================================================
    // 1st Pixel Processing (row = row_base)
    // =========================================================
    current_row = row_base;
    if (current_row < height) {
        result = val_m2 + val_m1 + val_0 + val_p1 + val_p2; // Vertical Sum

        // Add Horizontal Neighbors
        if (has_L2) result += in[current_row * width + (col - 2)];
        if (has_L1) result += in[current_row * width + (col - 1)];
        if (has_R1) result += in[current_row * width + (col + 1)];
        if (has_R2) result += in[current_row * width + (col + 2)];

        out[current_row * width + col] = result;
    }

    // =========================================================
    // 2nd Pixel Processing (row = row_base + 1)
    // =========================================================
    // [Shift Window Down]
    val_m2 = val_m1;
    val_m1 = val_0;
    val_0  = val_p1;
    val_p1 = val_p2;
    val_p2 = 0.0f; // Reset new bottom

    // Load new boundary (row + 2 relative to current, so row_base + 3)
    if (row_base + 3 < height) {
        val_p2 = in[(row_base + 3) * width + col];
    }

    current_row = row_base + 1;
    if (current_row < height) {
        result = val_m2 + val_m1 + val_0 + val_p1 + val_p2;

        if (has_L2) result += in[current_row * width + (col - 2)];
        if (has_L1) result += in[current_row * width + (col - 1)];
        if (has_R1) result += in[current_row * width + (col + 1)];
        if (has_R2) result += in[current_row * width + (col + 2)];

        out[current_row * width + col] = result;
    }

    // =========================================================
    // 3rd Pixel Processing (row = row_base + 2)
    // =========================================================
    // [Shift Window Down]
    val_m2 = val_m1;
    val_m1 = val_0;
    val_0  = val_p1;
    val_p1 = val_p2;
    val_p2 = 0.0f;

    // Load new boundary (row_base + 4)
    if (row_base + 4 < height) {
        val_p2 = in[(row_base + 4) * width + col];
    }

    current_row = row_base + 2;
    if (current_row < height) {
        result = val_m2 + val_m1 + val_0 + val_p1 + val_p2;

        if (has_L2) result += in[current_row * width + (col - 2)];
        if (has_L1) result += in[current_row * width + (col - 1)];
        if (has_R1) result += in[current_row * width + (col + 1)];
        if (has_R2) result += in[current_row * width + (col + 2)];

        out[current_row * width + col] = result;
    }

    // =========================================================
    // 4th Pixel Processing (row = row_base + 3)
    // =========================================================
    // [Shift Window Down]
    val_m2 = val_m1;
    val_m1 = val_0;
    val_0  = val_p1;
    val_p1 = val_p2;
    val_p2 = 0.0f;

    // Load new boundary (row_base + 5)
    if (row_base + 5 < height) {
        val_p2 = in[(row_base + 5) * width + col];
    }

    current_row = row_base + 3;
    if (current_row < height) {
        result = val_m2 + val_m1 + val_0 + val_p1 + val_p2;

        if (has_L2) result += in[current_row * width + (col - 2)];
        if (has_L1) result += in[current_row * width + (col - 1)];
        if (has_R1) result += in[current_row * width + (col + 1)];
        if (has_R2) result += in[current_row * width + (col + 2)];

        out[current_row * width + col] = result;
    }
  }
}

__global__ void stencil_2D_global_coarsen_4(const float* __restrict__ in, float* __restrict__ out, const int width, const int height) {
  // Global row and column indices
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_base = (blockIdx.y * blockDim.y + threadIdx.y) * 4; // Coarsening by factor of 2 in row direction
  const bool has_left = (col - 1) >= 0;
  const bool has_right = (col + 1) < width;

  // Process 2 pixels in the row direction
  int row = row_base;
  float result = 0.0f;
  float bottom = 0.0f, middle = 0.0f, top = 0.0f;

  if (col < width) {
    // 1st pixel (row, col)
    if (row - 1 >= 0 && row - 1 < height) {
      top = in[(row - 1) * width + col];
    }
    if (row >= 0 && row < height) {
      middle = in[row * width + col];
    }
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }

    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 2nd pixel (row + 1, col)
    row = row_base + 1;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }

    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 3rd pixel (row + 2, col)
    row = row_base + 2;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 4th pixel (row + 3, col)
    row = row_base + 3;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }
  }
}

__global__ void stencil_2D_global_coarsen_8(const float* __restrict__ in, float* __restrict__ out, const int width, const int height) {
  // Global row and column indices
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_base = (blockIdx.y * blockDim.y + threadIdx.y) * 8; // Coarsening by factor of 2 in row direction
  const bool has_left = (col - 1) >= 0;
  const bool has_right = (col + 1) < width;

  // Process 8 pixels in the row direction
  int row = row_base;
  float result = 0.0f;
  float bottom = 0.0f, middle = 0.0f, top = 0.0f;

  if (col < width) {
    // 1st pixel (row, col)
    if (row - 1 >= 0 && row - 1 < height) {
      top = in[(row - 1) * width + col];
    }
    if (row >= 0 && row < height) {
      middle = in[row * width + col];
    }
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }

    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 2nd pixel (row + 1, col)
    row = row_base + 1;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }

    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 3rd pixel (row + 2, col)
    row = row_base + 2;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 4th pixel (row + 3, col)
    row = row_base + 3;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }

    // 5th pixel (row + 4, col)
    row = row_base + 4;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }
    // 6th pixel (row + 5, col)
    row = row_base + 5;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }
    // 7th pixel (row + 6, col)
    row = row_base + 6;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
    }
    // 8th pixel (row + 7, col)
    row = row_base + 7;
    result = 0.0f;
    top = middle;
    middle = bottom;
    bottom = 0.0f;
    if (row + 1 >= 0 && row + 1 < height) {
      bottom = in[(row + 1) * width + col];
    }
    if (row < height) {
      result = top + middle + bottom;

      if (has_left) {
        result += in[row * width + (col - 1)];
      }
      if (has_right) {
        result += in[row * width + (col + 1)];
      }
      out[row * width + col] = result;
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

  gpuTKTime_start(Compute, "Performing CUDA computation");

  dim3 dimGrid((numColumns + TILE_WIDTH - 1) / TILE_WIDTH,
               (numRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  // Coarsened block: mapping 1 thread to 2 pixels in row direction.
  // dim3 dimBlockCoarsen2(TILE_WIDTH, TILE_WIDTH / 2, 1);
  // stencil_2D_global_coarsen_2<<<dimGrid, dimBlockCoarsen2>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);

  // Coarsened block: mapping 1 thread to 4 pixels in row direction.
  dim3 dimBlockCoarsen4(TILE_WIDTH, TILE_WIDTH / 4, 1);
#if RADIUS == 1
  stencil_2D_global_coarsen_4<<<dimGrid, dimBlockCoarsen4>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);
#elif RADIUS == 2
  stencil_2D_global_coarsen_4_radius_2<<<dimGrid, dimBlockCoarsen4>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);
#else
#error "Unsupported RADIUS value. Please define RADIUS as 1 or 2"
#endif

  // // Coarsened block: mapping 1 thread to 8 pixels in row direction.
  // dim3 dimBlockCoarsen4(TILE_WIDTH, TILE_WIDTH / 8, 1);
  // stencil_2D_global_coarsen_8<<<dimGrid, dimBlockCoarsen4>>>(deviceInput, deviceOutput, /* width= */ numColumns, /* height= */ numRows);

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
