#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=5GB
#SBATCH --job-name=strong_scaling_pyramid_v1_64
#SBATCH --output=../../output/strong_scaling/logs/strong_scaling_pyramid_v1_64.log

# Load GCC and MPI modules.
module load GCC OpenMPI 

export PKG_CONFIG_PATH=$SCRATCH/apps/opencv-4.9.0/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$SCRATCH/apps/opencv-4.9.0/lib64:$LD_LIBRARY_PATH

# Run pyramid v1 hnsw.
mpirun -n 64 ../../pyramid_hnsw_hnswlib_v1.1.out ../../../data/glove/glove.1024000.txt 1024000 300 10240 512 16 16 200 ../../output/strong_scaling/raw/strong_scaling_pyramid_v1_64.csv
