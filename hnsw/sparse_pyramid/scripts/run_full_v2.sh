#!/bin/bash

VERSION="_v2"
NUM_THREADS=32
CPUS_PER_TASK=32
MAX_TASKS_PER_NODE=128
SAMPLE_SIZE=80000
m=8000

for i in {0..0}; do
    NUM_OF_NODES=1
    TASKS=$((4**i))

    if (( TASKS * NUM_THREADS > MAX_TASKS_PER_NODE )); then
        NUM_OF_NODES=$(( (TASKS * NUM_THREADS + MAX_TASKS_PER_NODE - 1) / MAX_TASKS_PER_NODE ))
    fi

    TASKS_PER_NODE=$(( (TASKS + NUM_OF_NODES - 1) / NUM_OF_NODES )) 

    K=2
    while (( K <= 2 )); do
        K_SEARCH=1
        while (( K_SEARCH <= 4 * TASKS )); do
            sed "s|__NODES__|$NUM_OF_NODES|" "$SCRATCH/benchmarks/SpKNN/anns/hnsw/sparse_pyramid/scripts/full.sh" | \
            sed "s|__TASKS__|$TASKS|" | \
            sed "s|__TASKS_PER_NODE__|$TASKS_PER_NODE|" | \
            sed "s|__CPUS_PER_TASK__|$CPUS_PER_TASK|" | \
            sed "s|__SAMPLE_SIZE__|$SAMPLE_SIZE|" | \
            sed "s|__NUM_THREADS__|$NUM_THREADS|" | \
            sed "s|__VERSION__|$VERSION|" | \
            sed "s|__m__|$m|" | \
            sed "s|__K__|$K|" | \
            sed "s|__K_SEARCH__|$K_SEARCH|" > "$SCRATCH/benchmarks/SpKNN/anns/hnsw/sparse_pyramid/scripts/temp_full_${VERSION}_${TASKS}_${K}_${K_SEARCH}.sh"

            sbatch "$SCRATCH/benchmarks/SpKNN/anns/hnsw/sparse_pyramid/scripts/temp_full_${VERSION}_${TASKS}_${K}_${K_SEARCH}.sh"
            sleep 1

            K_SEARCH=$((K_SEARCH * 2))
        done
        K=$((K * 2))
    done
done
