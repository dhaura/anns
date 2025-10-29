#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --nodes=__NODES__
#SBATCH --ntasks-per-node=__TASKS_PER_NODE__
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --constraint=cpu
#SBATCH --output=logs/full__VERSION_____TASKS_____K_____K_SEARCH___%j.log

export OMP_NUM_THREADS=__CPUS_PER_TASK__

srun -N __NODES__ -n __TASKS__ --cpu-bind=cores $SCRATCH/benchmarks/SpKNN/anns/hnsw/sparse_pyramid/build/sparse_pyramid__VERSION__ $SCRATCH/datasets/SpKNN/grassRMA/base_full.csr __SAMPLE_SIZE__ __m__ __K__ __K_SEARCH__ 16 200 $SCRATCH/datasets/SpKNN/grassRMA/queries.dev.csr $SCRATCH/datasets/SpKNN/grassRMA/base_full.dev.gt
