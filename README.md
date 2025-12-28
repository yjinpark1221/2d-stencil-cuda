# Profile-driven Optimization: 2D Stencil Kernel

Term Project for Computing 2 (SNU GSDS)

## Generating Dataset

- Source code `src/generate_data.cpp`

### How to generate dataset

Run the following commands would compile the source code (executable file path: `build/generate_data`) and create dataset (dataset directory path: `Dataset/0/, Dataset/1/, ...`).
```
cd (DIRECTORY_PATH)/src
make generate_data
```

### How to adjust dataset size

To adjust the size of each dataset, you may change `std::vector<std::pair<int, int> > matrix_shapes` in `generate_data.cpp`. 

- Default sizes are 16 by 16 (`Dataset/0/`), 32 by 32(`Dataset/1/`), and 8192 by 8192(`Dataset/2/`).
- The experiments were run on 8192 by 8192 dataset. 

## Generating Executable files

Modify the following lines in `src/Makefile` to include and link GPU TK.
```
# GPU TK paths
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -lgputk -lcuda
LIBS= $(LIBgpuTK)
```

Run the following commands to compile and generate executable files.
(Requirement: nvcc and LIBgpuTK)
```
make all # to generate all
make baseline
make tiling
make coarsen
make hybrid
```

You may set your `TILE_WIDTH` and `RADIUS` as follows. (default sizes are `TILE_WIDTH`: 16, `RADIUS`: 1.)
```
make tiling TILE_WIDTH=32 RADIUS=2
```

## Running and Profiling

You may run your executable using the following command.
```
  (MY_BIN_PATH) \
    -e (MY_DATASET_DIR)/output.raw \
    -i (MY_DATASET_DIR)/input.raw \
    -o (MY_RESULT_PATH) \
    -t matrix
```
You may profile your kernel using using the following command. (Refer to NVIDIA documentation for more details.)
```
  TMPDIR=(MY_TMPDIR) ncu \
    --set full \
    -o (MY_REPORT_PATH) -f \
    (MY_BIN_PATH) \
    -e (MY_DATASET_DIR)/output.raw \
    -i (MY_DATASET_DIR)/input.raw \
    -o (MY_RESULT_PATH) \
    -t matrix
```

## Running by Scripts

### `scripts/run_generic.sh`

This shell script file can build, run, and profile a kernel for a dataset.

You may set the tile width, radius, and dataset number by command-line arguments.

For example, 
```
./run_generic.sh -type baseline -dataset 2 -tile 16 -radius 1
```

### `scripts/run_all.sh` 

This shell script file was used for submitting jobs to slurm workload manager of the GSDS server for Computing 2 class. 

It was originally designed to run all tests with a single job.

Because of the 1 minute time limit for computing nodes, I have uncommented one among the following lines and submitted  multiple jobs for the experiments in the report. 

```
./run_generic.sh -type baseline -dataset 2 -tile 16 -radius 1
./run_generic.sh -type tiling -dataset 2 -tile 16 -radius 1
./run_generic.sh -type coarsen -dataset 2 -tile 32 -radius 1
./run_generic.sh -type hybrid -dataset 2 -tile 32 -radius 1

./run_generic.sh -type baseline -dataset 2 -tile 16 -radius 2
./run_generic.sh -type tiling -dataset 2 -tile 16 -radius 2
./run_generic.sh -type coarsen -dataset 2 -tile 32 -radius 2
./run_generic.sh -type hybrid -dataset 2 -tile 32 -radius 2
```
## Code Explanation

Please refer to the report for more details.

- `baseline.cu` : no tiling, no coarsening
- `tiling.cu` : tiling (tile width is adjustable by setting TILE_WIDTH in Makefile or by command-line arguments of shell scripts.
- `coarsen.cu` : coarsened without tiling (default coarsening factor is set to 4 since it had the best perforamance among 2, 4, and 8)
- `hybrid.cu` : coarsened with tiling 
