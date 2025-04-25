#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=5GB
#SBATCH --job-name=strong_scaling_pyramid_v2_2
#SBATCH --output=../../output/strong_scaling/logs/strong_scaling_pyramid_v2_2.log

# Load GCC and MPI modules.
module load GCC OpenMPI 

export PKG_CONFIG_PATH=$SCRATCH/apps/opencv-4.9.0/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$SCRATCH/apps/opencv-4.9.0/lib64:$LD_LIBRARY_PATH

# Run pyramid v2 hnsw.
mpirun -n 2 ../../pyramid_hnsw_hnswlib_v2.1.out ../../../data/glove/glove.1024000.txt 1024000 300 10240 512 16 16 200 ../../output/strong_scaling/raw/strong_scaling_pyramid_v2_2.csv
 
