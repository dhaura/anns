#include <iostream>
#include <random>
#include <queue>
#include <vector>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <mpi.h>
#include "../../hnswlib/hnswlib/hnswlib.h"

static void read_txt(std::string filename, float* data, int input_size, int dimension) {
    
    std::ifstream infile(filename); // Open the file for reading

    if (infile) {
        std::string line;
        int i = 0;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            float value;
            int j = 0;

            // Read each value (label followed by features)
            while (iss >> value) {
                data[i * dimension + j] = value;
                j++;
            }
            i++;
        }
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_filepath> <input_size> <dimension> <M> <ef_construction> <num_threads> <query_input_filepath> <query_input_size>" << std::endl;
        return 1;
    }
    
    std::string input_filepath = argv[1];
    int input_size = std::stoi(argv[2]);
    int dimension = std::stoi(argv[3]);
    int M = std::stoi(argv[4]);
    int ef_construction = std::stoi(argv[5]);
    int p = std::stoi(argv[6]);
    std::string query_input_filepath = argv[7];
    int query_input_size = std::stoi(argv[8]);

    float* data = new float[input_size * dimension];

    int local_input_size = input_size / world_size;
    if (rank == world_size - 1) {
        local_input_size = input_size - (world_size - 1) * local_input_size;
    }
    float* local_data = new float[input_size * dimension];

    if (rank == 0) {
        read_txt(input_filepath, data, input_size, dimension);

        for (int i = 0; i < local_input_size; ++i) {
            for (int j = 0; j < dimension; ++j) {
                local_data[i * dimension + j] = data[i * dimension + j];
            }
        }

        // Distributing data to all processes
        for (int i = 1; i < world_size; ++i) {
            int start_index = i * local_input_size;
            int end_index = (i + 1) * local_input_size;
            if (i == world_size - 1) {
                end_index = input_size;
            }
            
            int num_elements = (end_index - start_index) * dimension;
            MPI_Send(data + start_index * dimension, num_elements, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        }
    } else {
        // Receiving data from process 0
        MPI_Recv(local_data, local_input_size * dimension, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    // Initing index
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, local_input_size, M, ef_construction);


    // Add data to index
    #pragma omp parallel for num_threads(p)
    for (int i = 0; i < local_input_size; i++) {
        alg_hnsw->addPoint(local_data + i * dimension, i);
    }

    hnswlib::BruteforceSearch<float>* alg_brute = new hnswlib::BruteforceSearch<float>(&space, input_size);

    if (rank == 0) {
        #pragma omp parallel for num_threads(p)
        for (int i = 0; i < input_size; i++) {
            alg_brute->addPoint(data + i * dimension, i);
        }
    }

    float* query_data = new float[query_input_size * dimension];
    read_txt(query_input_filepath, query_data, query_input_size, dimension);

    double correct = 0;
    // #pragma omp parallel for num_threads(p)
    for (int i = 0; i < query_input_size; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> local_result = alg_hnsw->searchKnn(query_data + i * dimension, 1);
        
        float distance = local_result.top().first;
        int label_id = local_result.top().second + rank * (input_size / world_size);
        
        std::vector<float> distances;
        std::vector<int> labels;
        if (rank == 0) {
            distances.resize(world_size);
            labels.resize(world_size);
        }
        MPI_Gather(&distance, 1, MPI_FLOAT, distances.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&label_id, 1, MPI_INT, labels.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);


        if (rank == 0) {
            int hnsw_id = labels[0];
            float min_distance = distances[0];
            for (int j = 1; j < world_size; ++j) {
                if (distances[j] < min_distance) {
                    min_distance = distances[j];
                    hnsw_id = labels[j];
                }
            }

            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_brute->searchKnn(query_data + i * dimension, 1);
            int brute_force_id = result.top().second;
            if (hnsw_id == brute_force_id) {
                correct++;
            }
        }
    }

    if (rank == 0) {
        float recall = correct / query_input_size;
        std::cout << "Recall: " << recall << "\n";
    }

    delete[] local_data;
    delete alg_hnsw;

    MPI_Finalize();
    return 0;
}
