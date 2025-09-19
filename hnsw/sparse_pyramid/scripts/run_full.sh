#!/bin/bash

for i in {0..3}; do
    NUM_OF_NODES=1
    TASKS_PER_NODE=$((2**i))
    CPUS_PER_TASK=32
    SAMPLE_SIZE=160000
    K=$((2**i))


    sed "s|__NODES__|$NUM_OF_NODES|" $SCRATCH/benchmarks/SpKNN/anns/hnsw/sparse_pyramid/scripts/full.sh | \
    sed "s|__TASKS_PER_NODE__|$TASKS_PER_NODE|" | \
    sed "s|__CPUS_PER_TASK__|$CPUS_PER_TASK|" | \
    sed "s|__SAMPLE_SIZE__|$SAMPLE_SIZE|" | \
    sed "s|__K__|$K|" > $SCRATCH/benchmarks/SpKNN/anns/hnsw/sparse_pyramid/scripts/temp_full_$TASKS_PER_NODE.sh

    sbatch $SCRATCH/benchmarks/SpKNN/anns/hnsw/sparse_pyramid/scripts/temp_full_$TASKS_PER_NODE.sh
    sleep 1
done
