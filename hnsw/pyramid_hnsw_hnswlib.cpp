#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include "../../hnswlib/hnswlib/hnswlib.h"

void read_txt(std::string filename, float* data, int input_size, int dimension) {
    
    std::ifstream infile(filename); // Open the file for reading.

    if (infile) {
        std::string line;
        int i = 0;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            float value;
            int j = 0;

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

void sample_input(float* datamatrix, int input_size, int dim,
                   int num_samples, cv::Mat& sampled_data) {

    srand(time(0));

    sampled_data = cv::Mat(num_samples, dim, CV_32F);

    std::vector<int> indices(input_size);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());

    for (int i = 0; i < num_samples; ++i) {
        int idx = indices[i];
        for (int j = 0; j < dim; ++j) {
            sampled_data.at<float>(i, j) = datamatrix[idx * dim + j];
        }
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 11) {
        std::cerr << "Usage: " << argv[0] << " <input_filepath> <input_size> <dimension> <sample_size> <M> <ef_construction> <num_threads> <randomize_input> <query_input_filepath> <query_input_size>" << std::endl;
        return 1;
    }
    
    // Parse command line arguments into variables.
    std::string input_filepath = argv[1];
    int input_size = std::stoi(argv[2]);
    int dimension = std::stoi(argv[3]);
    int sample_size = std::stoi(argv[4]);
    int M = std::stoi(argv[5]);
    int ef_construction = std::stoi(argv[6]);
    int p = std::stoi(argv[7]);
    bool randomize_input = std::stoi(argv[8]);
    std::string query_input_filepath = argv[9];
    int query_input_size = std::stoi(argv[10]);

    float* data = new float[input_size * dimension];
    float* local_data;
    int local_input_size;

    read_txt(input_filepath, data, input_size, dimension);

    // Number of clusters
    int k;

    // Output labels and centers
    cv::Mat labels;
    cv::Mat centers;

    float* m_centers;

    hnswlib::L2Space meta_space(dimension);
    hnswlib::HierarchicalNSW<float>* meta_hnsw;

    if (rank == 0) {
        cv::Mat sampled_data;
        sample_input(data, input_size, dimension, sample_size, sampled_data);

        k = world_size;

        // kmeans flags
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.1);
        int attempts = 3;
        int flags = cv::KMEANS_PP_CENTERS;

        // Run kmeans
        cv::kmeans(sampled_data, k, labels, criteria, attempts, flags, centers);

        m_centers = centers.ptr<float>();
        meta_hnsw = new hnswlib::HierarchicalNSW<float>(&meta_space, k, M, ef_construction);

        for (int i = 0; i < k; i++) {
            meta_hnsw->addPoint(m_centers + i * dimension, i);
        }

        // TODO: divide m centers into w groups -> w = world_size
        std::vector<std::vector<float*>> data_to_send(world_size);
        for (int i = 0; i < input_size; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = meta_hnsw->searchKnn(data + i * dimension, 1);
            float distance = result.top().first;
            int label_id = result.top().second;

            data_to_send[label_id].push_back(data + i * dimension);
        }

        // Allocate local data for rank 0.
        local_input_size = data_to_send[0].size();
        local_data = new float[local_input_size * dimension];
        float** data_ptrs = data_to_send[0].data();
        for (int i = 0; i < local_input_size; ++i) {
            std::memcpy(local_data + i * dimension, data_ptrs[i], dimension * sizeof(float));
        }

        for (int i = 1; i < world_size; i++) {
            int data_size_to_send_i = data_to_send[i].size();
            float** data_ptrs = data_to_send[i].data();
            float* data_to_send_i = new float[data_size_to_send_i * dimension];

            for (int j = 0; j < data_size_to_send_i; ++j) {
                std::memcpy(data_to_send_i + j * dimension, data_ptrs[j], dimension * sizeof(float));
            }

            MPI_Send(&data_size_to_send_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(data_to_send_i, data_size_to_send_i * dimension, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Receiving data from process 0.
        MPI_Recv(&local_input_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_data = new float[local_input_size * dimension];
        MPI_Recv(local_data, local_input_size * dimension, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int label_start = 0;
    MPI_Exscan(&local_input_size, &label_start, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Initiate hnsw index.
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float>* local_hnsw = new hnswlib::HierarchicalNSW<float>(&space, local_input_size, M, ef_construction);

    // Add data to hnsw index.
    for (int i = 0; i < local_input_size; i++) {
        local_hnsw->addPoint(local_data + i * dimension, label_start + i);
    }

    int local_query_size = 0;
    float* local_query_data = nullptr;
    if (rank == 0) {
        float* query_data = new float[query_input_size * dimension];
        read_txt(query_input_filepath, query_data, query_input_size, dimension);

        std::vector<std::vector<float*>> query_to_send(world_size);
        for (int i = 0; i < query_input_size; ++i) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = meta_hnsw->searchKnn(query_data + i * dimension, 1);
            float distance = result.top().first;
            int label_id = result.top().second;

            query_to_send[label_id].push_back(query_data + i * dimension);
        }

        // Allocate local data for rank 0.
        local_query_size = query_to_send[0].size();
        local_query_data = new float[local_query_size * dimension];
        float** query_ptrs = query_to_send[0].data();
        for (int i = 0; i < local_query_size; ++i) {
            std::memcpy(local_query_data + i * dimension, query_ptrs[i], dimension * sizeof(float));
        }


        for (int i = 1; i < world_size; i++) {
            int query_size_to_send_i = query_to_send[i].size();
            float** query_ptrs = query_to_send[i].data();
            float* query_to_send_i = new float[query_size_to_send_i * dimension];

            for (int j = 0; j < query_size_to_send_i; ++j) {
                std::memcpy(query_to_send_i + j * dimension, query_ptrs[j], dimension * sizeof(float));
            }

            MPI_Send(&query_size_to_send_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(query_to_send_i, query_size_to_send_i * dimension, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }

    } else {
        // Receiving data from process 0.
        MPI_Recv(&local_query_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_query_data = new float[local_query_size * dimension];
        MPI_Recv(local_query_data, local_query_size * dimension, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Print received query data
    std::vector<int> local_query_results(local_query_size);
    for (int i = 0; i < local_query_size; ++i) {
        float* query = local_query_data + i * dimension;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = local_hnsw->searchKnn(query, 1);
        float distance = result.top().first;
        int label_id = result.top().second;
        local_query_results[i] = label_id;
    }

    hnswlib::BruteforceSearch<float>* alg_brute = new hnswlib::BruteforceSearch<float>(&space, input_size);
    for (int i = 0; i < input_size; i++) {
        alg_brute->addPoint(data + i * dimension, i);
    }

    float correct = 0;
    for (int i = 0; i < local_query_size; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_brute->searchKnn(local_query_data + i * dimension, 1);
        int brute_force_id = result.top().second;
        if (local_query_results[i] == brute_force_id) {
            correct++;
        }
    }

    if (rank == 0) {
        for (int i = 1; i < world_size; i++) {
            float correct_i;
            MPI_Recv(&correct_i, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            correct += correct_i;
        }
        float recall = correct / query_input_size;
        std::cout << "Recall: " << recall << std::endl;
    } else {
        MPI_Send(&correct, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    
    return 0;
}
