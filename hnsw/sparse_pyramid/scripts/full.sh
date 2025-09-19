#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=10:30:00
#SBATCH --nodes=__NODES__
#SBATCH --ntasks-per-node=__TASKS_PER_NODE__
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=__CPUS_PER_TASK__

srun -n __TASKS_PER_NODE__ --cpu-bind=cores ./build/sparse_pyramid $SCRATCH/datasets/SpKNN/grassRMA/base_full.csr __SAMPLE_SIZE__ 12 __K__ 16 200 $SCRATCH/datasets/SpKNN/grassRMA/queries.dev.csr $SCRATCH/datasets/SpKNN/grassRMA/base_full.dev.gt
