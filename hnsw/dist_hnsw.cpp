#include <vector>
#include <unordered_set>
#include <map>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <mpi.h>

template <typename VALUE_TYPE>
using ValueType2DVector = std::vector<std::vector<VALUE_TYPE>>;

struct Node {
    
    int id;
    int node_max_level;
    std::vector<float> feature_vector;
    std::vector<std::vector<std::pair<int, float>>> neighbors;
};

struct HNSW {
    
    int max_level = -1;
    int entry_point = -1;
    std::map<int, Node> nodes;
};

static void read_txt(std::string filename, ValueType2DVector<float>* datamatrix) {
    
    std::ifstream infile(filename); // Open the file for reading.
    std::vector<std::vector<float>> data; // Vector to hold the loaded data.

    if (infile) {
        std::string line;
        while (std::getline(infile, line)) {
            std::vector<float> row;
            std::istringstream iss(line);
            float value;

            // Read each value (label followed by features).
            while (iss >> value) {
                row.push_back(value);
            }

            // Add the row (data vector) to the data vector.
            data.push_back(row);
        }
        datamatrix->resize(data.size());
        for(int i = 0; i < data.size(); i++){
            (*datamatrix)[i] = data[i];
        }
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

std::vector<std::pair<int, float>> find_top_K(std::vector<std::pair<int, float>>& candidates, int k) {
    
    std::vector<std::pair<int, float>> top_k = std::vector<std::pair<int, float>>(candidates.begin(), candidates.end());
    std::sort(top_k.begin(), top_k.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });
    if (top_k.size() > k) {
        top_k.resize(k);
    }
    return top_k;
}

float find_min(std::vector<std::pair<int, float>>& nodes) {
    
    float min_distance = std::numeric_limits<float>::max();
    for (const auto& node : nodes) {
        if (node.second < min_distance) {
            min_distance = node.second;
        }
    }
    return min_distance;
}

float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float calculateIntersection(const std::vector<int>& hnsw_ids, const std::vector<int>& brute_force_ids, int k) {
    
    std::unordered_set<int> hnsw_set(hnsw_ids.begin(), hnsw_ids.end());

    int intersection_count = std::accumulate(brute_force_ids.begin(), brute_force_ids.end(), 0, 
        [&hnsw_set](int count, int id) {
            return count + (hnsw_set.count(id) > 0);
        });

    return static_cast<float>(intersection_count) / k;
}

float calculate_recall(std::vector<std::pair<int, float>>& hnsw_results, std::vector<float>& query_vector, ValueType2DVector<float>& datamatrix, int k) {

    std::vector<std::pair<int, float>> brute_force_results;
    for (int i = 0; i < datamatrix.size(); ++i) {
        float distance = euclidean_distance(query_vector, datamatrix[i]);
        brute_force_results.push_back(std::make_pair(i, distance));
    }
    brute_force_results = find_top_K(brute_force_results, k);

    std::vector<int> brute_force_ids;
    for (const auto& result : brute_force_results) {
        brute_force_ids.push_back(result.first);
    }

    std::vector<int> hnsw_ids;
    for (const auto& result : hnsw_results) {
        hnsw_ids.push_back(result.first);
    }
    
    return calculateIntersection(hnsw_ids, brute_force_ids, k);
}

std::vector<std::pair<int, float>> search_level(HNSW& hnsw, int level, std::vector<float> query_feature_vec, int factor, std::vector<std::pair<int, float>>& candidates) {
    
    std::vector<std::pair<int, float>> winners = std::vector<std::pair<int, float>>(candidates.begin(), candidates.end());
    std::vector<int> visited_node_ids;

    while (candidates.size() > 0 && find_min(candidates) <= find_min(winners)) {
        Node current_node = hnsw.nodes[candidates.back().first];
        candidates.pop_back();
        visited_node_ids.push_back(current_node.id);

        for (const auto& _neighbor : current_node.neighbors[level]) {
            int neighbor = _neighbor.first;
            if (std::find(visited_node_ids.begin(), visited_node_ids.end(), neighbor) != visited_node_ids.end()) {
                continue;
            }
            visited_node_ids.push_back(neighbor);

            float distance = euclidean_distance(query_feature_vec, hnsw.nodes[neighbor].feature_vector);
            candidates.push_back(std::make_pair(neighbor, distance));
            winners.push_back(std::make_pair(neighbor, distance));
        }
        winners = find_top_K(winners, factor);
    }
    return winners; 
}

std::vector<std::pair<int, float>> query_hnsw(HNSW& hnsw, std::vector<float> query_feature_vec, int k, int l) {
    
    std::vector<std::pair<int, float>> results;
    std::vector<std::pair<int, float>> candidates;
    candidates.push_back(std::make_pair(hnsw.entry_point, euclidean_distance(query_feature_vec, hnsw.nodes[hnsw.entry_point].feature_vector)));

    for (int j = hnsw.max_level; j > 0; --j) {
        candidates = search_level(hnsw, j, query_feature_vec, 1, candidates);
    }
    candidates = search_level(hnsw, 0, query_feature_vec, l, candidates);
    results = find_top_K(candidates, k);
    return results;
}

void build_hnsw(HNSW& hnsw, int input_size, ValueType2DVector<float>& datamatrix, int l, int M, int rank, int world_size) {

    int local_input_size = input_size / world_size;
    for (int i = 0; i < datamatrix.size(); ++i) {
        const auto& feature_vec = datamatrix[i];

        Node* node = new Node();
        node->id = i + rank * local_input_size;
        node->feature_vector = feature_vec;  

        if (hnsw.entry_point == -1) {
            node->node_max_level = hnsw.max_level;
            node->neighbors.resize(hnsw.max_level + 1);
            hnsw.entry_point = node->id;
            hnsw.nodes[node->id] = *node;
            continue;
        }

        int level = rand() % (hnsw.max_level + 1);
        node->node_max_level = level;
        node->neighbors.resize(level + 1);
        std::vector<std::pair<int, float>> candidates;
        candidates.push_back(std::make_pair(hnsw.entry_point, euclidean_distance(feature_vec, hnsw.nodes[hnsw.entry_point].feature_vector)));

        for (int j = hnsw.max_level; j > level; --j) {
            candidates = search_level(hnsw, j, feature_vec, 1, candidates);
        }

        for (int j = level; j >= 0; --j) {
            candidates = search_level(hnsw, j, feature_vec, l, candidates);
            for (const auto& candidate : find_top_K(candidates, M)) {
                node->neighbors[j].push_back(std::make_pair(candidate.first, candidate.second));

                std::vector<std::pair<int, float>> candidate_neighbors = hnsw.nodes[candidate.first].neighbors[j];
                candidate_neighbors.push_back(std::make_pair(node->id, candidate.second));
                hnsw.nodes[candidate.first].neighbors[j] = find_top_K(candidate_neighbors, M);
            }
            candidates = find_top_K(candidates, 1);
        }
        hnsw.nodes[node->id] = *node;
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_filepath> <input_size> <dimension> <k> <num_of_levels> <l> <M> <query_inpuy_file_path>" << std::endl;
        return 1;
    }
    
    std::string input_filepath = argv[1];
    int input_size = std::stoi(argv[2]);
    int dimension = std::stoi(argv[3]);
    int k = std::stoi(argv[4]);
    int num_of_levels = std::stoi(argv[5]);
    int l = std::stoi(argv[6]);
    int M = std::stoi(argv[7]);
    std::string query_input_filepath = argv[8];

    ValueType2DVector<float> datamatrix;
    ValueType2DVector<float> local_datamatrix;
    int local_input_size = input_size / world_size;
    if (rank == world_size - 1) {
        local_input_size = input_size - (world_size - 1) * local_input_size;
    }

    if (rank == 0) {
        read_txt(input_filepath, &datamatrix);

        local_datamatrix.resize(local_input_size);
        for (int i = 0; i < local_input_size; ++i) {
            local_datamatrix[i] = datamatrix[i];
        }

        // std::random_device rd;
        // std::mt19937 g(rd());
        // std::shuffle(datamatrix.begin(), datamatrix.end(), g);
        
        for (int i = 1; i < world_size; ++i) {
            int start_index = i * local_input_size;
            int end_index = (i + 1) * local_input_size;
            if (i == world_size - 1) {
                end_index = input_size;
            }
           
            std::vector<float> flattened_data;
            for (int j = start_index; j < end_index; ++j) {
                flattened_data.insert(flattened_data.end(), datamatrix[j].begin(), datamatrix[j].end());
            }

            int num_elements = flattened_data.size();
            MPI_Send(flattened_data.data(), num_elements, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        }
    } else {
        int num_elements = dimension * local_input_size;
        
        std::vector<float> flattened_data_recv(num_elements);
        MPI_Recv(flattened_data_recv.data(), num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_datamatrix.resize(local_input_size);
        int index = 0;
        for (int i = 0; i < local_input_size; ++i) {
            local_datamatrix[i].assign(flattened_data_recv.begin() + index, flattened_data_recv.begin() + index + dimension);
            index += dimension;
        }
    }

    HNSW hnsw;
    hnsw.max_level = num_of_levels - 1;

    build_hnsw(hnsw, input_size, local_datamatrix, l, M, rank, world_size);

    MPI_Barrier(MPI_COMM_WORLD);

    ValueType2DVector<float> query_datamatrix;
    read_txt(query_input_filepath, &query_datamatrix);

    std::vector<float> recalls;

    for (auto& query : query_datamatrix) {
        std::vector<std::pair<int, float>> local_results = query_hnsw(hnsw, query, k, l);
        std::vector<std::pair<int, float>> results(k * world_size);
        MPI_Gather(local_results.data(), k * sizeof(std::pair<int, float>), MPI_BYTE, results.data(), k * sizeof(std::pair<int, float>), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::vector<std::pair<int, float>> final_results = find_top_K(results, k);
            float recall = calculate_recall(final_results, query, datamatrix, k);
            recalls.push_back(recall);
            std::cout << "Recall: " << recall << std::endl;
        }
    }
    
    if (rank == 0) {
        float mean_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
        std::cout << "Mean Recall: " << mean_recall << std::endl;
        std::cout << euclidean_distance({5.9, 3.0, 5.1, 1.8}, {0.9, 0.2, 1.0, 4.5}) << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

