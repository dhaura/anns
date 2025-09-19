#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=cpu
#SBATCH --output=%j.log

srun -n 2 ./build/sparse_pyramid $SCRATCH/datasets/SpKNN/grassRMA/base_small.csr 1600 12 4 16 200 $SCRATCH/datasets/SpKNN/grassRMA/queries.dev.csr $SCRATCH/datasets/SpKNN/grassRMA/base_small.dev.gt
