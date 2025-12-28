# Makefile for 2D Stencil Computation Examples
# Author: Yeonjin Park

# NVCC compiler
NVCC=nvcc
NVCC_FLAGS= -std=c++11 -O3 -lineinfo

# C++ compiler
GPP=g++
GPP_FLAGS= -std=c++11 -O3

# GPU TK paths
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -lgputk -lcuda
LIBS= $(LIBgpuTK)

TILE_WIDTH?=16
RADIUS?=1

OUTPUT_DIR?=./build

# Main targets (not including data generation)
all: baseline tiling coarsen hybrid

generate_data:
	mkdir -p $(OUTPUT_DIR)
	$(GPP) $(GPP_FLAGS) generate_data.cpp -o $(OUTPUT_DIR)/generate_data
	./$(OUTPUT_DIR)/generate_data

baseline:
	mkdir -p $(OUTPUT_DIR)
	$(NVCC) $(NVCC_FLAGS) baseline.cu $(INCLUDE) $(LIBS) -o $(OUTPUT_DIR)/baseline -DTILE_WIDTH=$(TILE_WIDTH) -DRADIUS=$(RADIUS)

tiling:
	mkdir -p $(OUTPUT_DIR)
	$(NVCC) $(NVCC_FLAGS) tiling.cu $(INCLUDE) $(LIBS) -o $(OUTPUT_DIR)/tiling -DTILE_WIDTH=$(TILE_WIDTH) -DRADIUS=$(RADIUS)

# output_tiling:
# 	$(NVCC) $(NVCC_FLAGS) output_tiling.cu $(INCLUDE) $(LIBS) -o $(OUTPUT_DIR)/output_tiling -DTILE_WIDTH=$(TILE_WIDTH) -DRADIUS=$(RADIUS)

coarsen:
	mkdir -p $(OUTPUT_DIR)
	$(NVCC) $(NVCC_FLAGS) coarsen.cu $(INCLUDE) $(LIBS) -o $(OUTPUT_DIR)/coarsen -DTILE_WIDTH=$(TILE_WIDTH) -DRADIUS=$(RADIUS)

hybrid:
	mkdir -p $(OUTPUT_DIR)
	$(NVCC) $(NVCC_FLAGS) hybrid.cu $(INCLUDE) $(LIBS) -o $(OUTPUT_DIR)/hybrid -DTILE_WIDTH=$(TILE_WIDTH) -DRADIUS=$(RADIUS)

clean:
	rm -f $(OUTPUT_DIR)/*

