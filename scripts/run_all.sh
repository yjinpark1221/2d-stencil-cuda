#!/bin/bash

#SBATCH --job-name=project
#SBATCH --output=project.log
#SBATCH --partition=class3
#SBATCH --gres=gpu:1
source ~/miniconda3/etc/profile.d/conda.sh
export PATH=$PATH:/usr/local/cuda-12.2/bin
conda activate comp2

echo "========================================="
echo " System Information                      "
echo "========================================="
nvidia-smi
echo "========================================="

cd ~/2d-stencil-cuda/src
# make clean

cd ../scripts
# ./run_generic.sh -type baseline -dataset 2 -tile 16 -radius 1
# ./run_generic.sh -type tiling -dataset 2 -tile 16 -radius 1
# ./run_generic.sh -type coarsen -dataset 2 -tile 32 -radius 1
# ./run_generic.sh -type hybrid -dataset 2 -tile 32 -radius 1


# The following executions output "The solution is NOT correct" since there are the output.raw in Dataset/ are of RADIUS 1.
# ./run_generic.sh -type baseline -dataset 2 -tile 16 -radius 2
# ./run_generic.sh -type tiling -dataset 2 -tile 16 -radius 2
# ./run_generic.sh -type coarsen -dataset 2 -tile 32 -radius 2
# ./run_generic.sh -type hybrid -dataset 2 -tile 32 -radius 2

echo "========================================="
echo " All Tasks Finished                      "
echo "========================================="
~
