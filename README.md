# ANNS
## HNSW

### 1. Shared Memory Parallelized HNSW
#### 1.1 Custom
1. Compile the code.
```bash
g++ ./hnsw.cpp -o hnsw.out -fopenmp
```
2. Run the code.
```bash
./hnsw.out <input_file_path> <input_size> <dimension> <num_of_levels> <l> <M> <num_threads> <query_input_filepath> 
```
```bash
./hnsw.out ../data/iris_dataset/iris.data.txt 150 4 5 12 13 1 ../data/iris_dataset/query/iris_query_points_1.txt 
```
#### 1.2 hnswlib
1. Compile the code.
```bash
g++ ./hnsw_hnswlib.cpp -o hnsw_hnswlib.out -fopenmp
```
2. Run the code.
```bash
./hnsw_hnswlib.out <input_filepath> <input_size> <dimension> <M> <ef_construction> <num_threads> <query_input_filepath> <query_input_size>
```
```bash
./hnsw_hnswlib.out ../data/iris_dataset/iris.data.txt 150 4 16 200 2 ../data/iris_dataset/query/iris_query_points_1.txt 11
```

### 2. Distributed HNSW (Naive)
#### 2.1 Custom
1. Compile the code.
```bash
mpic++ dist_hnsw.cpp -o dist_hnsw.out -fopenmp
```
2. Run the code.
```bash
mpirun -n <num_of_nodes> ./dist_hnsw.out <input_filepath> <input_size> <dimension> <num_of_levels> <l> <M> <num_threads> <randomize_input> <query_inpuy_file_path>
```
```bash
mpirun -n 4 ./dist_hnsw.out ../data/iris_dataset/iris.data.txt 150 4 5 12 15 2 0 ../data/iris_dataset/query/iris_query_points_1.txt
```
#### 2.2 hnswlib
1. Compile the code.
```bash
mpic++ dist_hnsw_hnswlib.cpp -o dist_hnsw_hnswlib.out -fopenmp
```
2. Run the code.
```bash
mpirun -n <num_of_nodes> ./dist_hnsw_hnswlib.out <input_filepath> <input_size> <dimension> <M> <ef_construction> <num_threads> <randomize_input> <query_input_filepath> <query_input_size>
```
```bash
mpirun -n 4 ./dist_hnsw_hnswlib.out ../data/iris_dataset/iris.data.txt 150 4 16 200 2 0 ../data/iris_dataset/query/iris_query_points_1.txt 11
```

### 3. Distributed HNSW (Pyramid Approach)
#### 2.1 hnswlib
1. Compile the code.
```bash
mpic++ pyramid_hnsw_hnswlib.cpp -o pyramid_hnsw_hnswlib.out -fopenmp `pkg-config --cflags --libs opencv4`
```
2. Run the code.
```bash
mpirun -n <num_of_nodes> ./pyramid_hnsw_hnswlib.out . <input_filepath> <input_size> <dimension> <sample_size> <m> <branching_factor> <M> <ef_construction> <num_threads> <randomize_input> <query_input_filepath> <query_input_size>
```
```bash
mpirun -n 4 ./pyramid_hnsw_hnswlib.out ../data/iris_dataset/iris.data.txt 150 4 30 12 2 16 200 2 0 ../data/iris_dataset/query/iris_query_points_1.txt 11
```
