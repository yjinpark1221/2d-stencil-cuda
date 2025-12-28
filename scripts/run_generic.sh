#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 -type <baseline|tiling|coarsen|hybrid>"
  exit 1
}

impl_type=""
dataset="2" # Default dataset
TILE_WIDTH=16
RADIUS=1

# Parse command-line arguments
while [ $# -gt 0 ]; do
  case "$1" in
    -type)
      impl_type="$2"
      shift 2
      ;;
    -dataset)
      dataset="$2"
      shift 2
      ;;
    -tile)
      TILE_WIDTH="$2"
      shift 2
      ;;
    -radius)
      RADIUS="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

# Validate required arguments
if [[ -z "$impl_type" ]]; then
  echo "Error: -type is required."
  usage
fi

# Initialize directories
base_dir=$(dirname "$0")/..
dataset_dir="${base_dir}/Dataset"
my_tmp_dir="${base_dir}/tmp_ncu_lock"
result_dir="${base_dir}/results/output/${impl_type}"
profile_dir="${base_dir}/results/profile/${impl_type}"

# Make directories if they do not exist
mkdir -p "${result_dir}"
mkdir -p "${my_tmp_dir}"
mkdir -p "${profile_dir}"

# Build binary
echo ">> Building binary for type: ${impl_type}"
cd "${base_dir}/sources"
make clean
make "${impl_type}" TILE_WIDTH="${TILE_WIDTH}" RADIUS="${RADIUS}"

bin_path="${base_dir}/sources/build/${impl_type}"

if [[ ! -f "$bin_path" ]]; then
    echo "Error: Binary not found at $bin_path"
    exit 1
fi

echo "========================================"
echo ">> Processing Dataset ${dataset} with ${impl_type}"
echo "========================================"

unset CUDA_VISIBLE_DEVICES

# Run the binary
echo ">> Running..."

"$bin_path" \
-e "${dataset_dir}/${dataset}/output.raw" \
-i "${dataset_dir}/${dataset}/input.raw" \
-o "${result_dir}/${dataset}.raw" \
-t matrix

# Run the profiler
echo ">> Running Profiler..."

TMPDIR="${my_tmp_dir}" ncu \
--set full \
-o "${profile_dir}/report_${dataset}_${TILE_WIDTH}w_${RADIUS}r" \
"$bin_path" \
-e "${dataset_dir}/${dataset}/output.raw" \
-i "${dataset_dir}/${dataset}/input.raw" \
-o "${result_dir}/${dataset}_profiled.raw" \
-t matrix

echo ">> Done Dataset ${dataset}"

echo ">> All tasks for ${impl_type} finished successfully."
