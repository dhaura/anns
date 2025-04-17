#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <omp.h>
#include "../../hnswlib/hnswlib/hnswlib.h"

template <typename VALUE_TYPE>
using ValueType2DVector = std::vector<std::vector<VALUE_TYPE>>;

MPI_Datatype FLOAT_DATAMATRIX;

int read_txt(std::string filename, ValueType2DVector<float> *datamatrix,
             int no_of_datapoints, int dim, int rank, int world_size, int offset = 0)
{

    std::ifstream infile(filename);       // Open the file for reading
    std::vector<std::vector<float>> data; // Vector to hold the loaded data

    if (!infile.is_open())
    {
        // Handle file opening error
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return -1;
    }
    std::cout << "Rank " << rank << " opened file " << filename << std::endl;

    int chunk_size = no_of_datapoints / world_size;
    int start_idx = rank * chunk_size;
    int end_index = 0;

    if (rank < world_size - 1)
    {
        end_index = (rank + 1) * chunk_size - 1;
    }
    else if (rank == world_size - 1)
    {
        end_index = std::min((rank + 1) * chunk_size - 1, no_of_datapoints - 1);
        chunk_size = no_of_datapoints - rank * chunk_size;
    }

    if (chunk_size == -1)
    {
        chunk_size = no_of_datapoints - start_idx;
    }

    std::cout << "Rank " << rank << " selected chunk size " << chunk_size << " starting " << start_idx << std::endl;

    // Skip lines up to start_idx
    std::string line;
    for (int i = 0; i < start_idx + offset && std::getline(infile, line); ++i)
        ;

    for (int i = 0; i < chunk_size; ++i)
    {
        if (std::getline(infile, line))
        {
            std::vector<float> row;
            std::istringstream iss(line);
            float value;

            // Read each value (label followed by features)
            while (iss >> value)
            {
                row.push_back(value);
            }
            data.push_back(row);
        }

        // Add the row (data vector) to the data vector
    }
    datamatrix->resize(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        (*datamatrix)[i] = data[i];
    }

    return chunk_size;
}

void sample_input(const std::vector<std::vector<float>> &datamatrix, int input_size, int dim,
                  int num_samples, float *&sampled_data)
{

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());

    // Create a vector to store the indices
    std::vector<int> indices(input_size);
    std::iota(indices.begin(), indices.end(), 0); // Fill indices from 0 to input_size-1

    // Shuffle the indices randomly
    std::shuffle(indices.begin(), indices.end(), rng);

    // Allocate memory for the 1D float array
    sampled_data = new float[num_samples * dim];

    // Fill the sampled_data array
    for (int i = 0; i < num_samples; ++i)
    {
        int idx = indices[i]; // Get the index of the sampled row
        // Copy the entire row from datamatrix[idx] directly to sampled_data
        std::memcpy(sampled_data + i * dim, datamatrix[idx].data(), dim * sizeof(float));
    }
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 13)
    {
        std::cerr << "Usage: " << argv[0] << " <input_filepath> <input_size> <dimension> <sample_size> <m> <branching_factor> <M> <ef_construction> <num_threads> <randomize_input> <query_input_filepath> <query_input_size>" << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
    std::string input_filepath = argv[1];
    int input_size = std::stoi(argv[2]);
    int dimension = std::stoi(argv[3]);
    int sample_size = std::stoi(argv[4]);
    int m = std::stoi(argv[5]);
    int k = std::stoi(argv[6]);
    int M = std::stoi(argv[7]);
    int ef_construction = std::stoi(argv[8]);
    int p = std::stoi(argv[9]);
    bool randomize_input = std::stoi(argv[10]);
    std::string query_input_filepath = argv[11];
    int query_input_size = std::stoi(argv[12]);

    ValueType2DVector<float> datamatrix;
    int chunk_size;
    chunk_size = read_txt(input_filepath, &datamatrix, input_size, dimension, rank, world_size);

    hnswlib::L2Space meta_space(dimension);
    hnswlib::HierarchicalNSW<float>* meta_hnsw;
    std::string meta_hnsw_path = "meta_hnsw.bin";

    float *sampled_data = new float[sample_size * dimension];
    sample_input(datamatrix, chunk_size, dimension, sample_size, sampled_data);

    int global_sample_size = sample_size * world_size;
    float *global_sampled_data = new float[global_sample_size * dimension];
    MPI_Gather(sampled_data, sample_size * dimension, MPI_FLOAT,
        global_sampled_data, sample_size * dimension, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cv::Mat sampled_data(global_sample_size, dimension, CV_32F, global_sampled_data);

        int w = world_size;

        // Output labels and centers.
        cv::Mat labels;
        cv::Mat centers;

        float *m_centers;

        // kmeans flags
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.1);
        int attempts = 3;
        int flags = cv::KMEANS_PP_CENTERS;

        // Run kmeans
        cv::kmeans(sampled_data, m, labels, criteria, attempts, flags, centers);

        meta_hnsw = new hnswlib::HierarchicalNSW<float>(&meta_space, m, M, ef_construction);
        #pragma omp parallel for num_threads(p)
        for (int i = 0; i < k; i++) {
            meta_hnsw->addPoint(centers.ptr<float>(i), i);
        }

        meta_hnsw->saveIndex(meta_hnsw_path);
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        meta_hnsw = new hnswlib::HierarchicalNSW<float>(&meta_space, meta_hnsw_path);
    }

    MPI_Finalize();

    return 0;
}
