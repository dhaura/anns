#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi.h>
#include <hnswlib.h>
#include <csr_matrix.h>

using Float2DVector = std::vector<std::vector<float>>;
using Float2DPairVector = std::vector<std::pair<int, std::vector<float>>>;

CSRMatrix *read_csr(const std::string &filename, int rank, int world_size)
{

    int64_t n_rows = 0, n_cols = 0, nnz = 0;

    std::ifstream csr_file(filename, std::ios::binary);
    csr_file.read(reinterpret_cast<char *>(&n_rows), sizeof(int64_t));
    csr_file.read(reinterpret_cast<char *>(&n_cols), sizeof(int64_t));
    csr_file.read(reinterpret_cast<char *>(&nnz), sizeof(int64_t));

    // Row partition.
    int64_t row_start = rank * (n_rows / world_size);
    int64_t row_end = (rank == world_size - 1) ? (n_rows - 1) : ((rank + 1) * (n_rows / world_size) - 1);
    int64_t local_n_rows = row_end - row_start + 1;

    // Load the full indptr range (only part needed).
    std::vector<int64_t> local_indptr(local_n_rows + 1);

    size_t indptr_offset = sizeof(int64_t) * 3 + sizeof(int64_t) * row_start;
    csr_file.seekg(indptr_offset);
    csr_file.read(reinterpret_cast<char *>(local_indptr.data()), sizeof(int64_t) * (local_n_rows + 1));

    // Compute data range from local indptr.
    int64_t csr_start = local_indptr[0];
    int64_t csr_end = local_indptr.back();
    int64_t local_nnz = csr_end - csr_start;

    // Shift local_indptr so that it starts from 0.
    for (auto &ptr : local_indptr)
        ptr -= csr_start;

    // Read indices.
    std::vector<int32_t> local_indices(local_nnz);
    size_t indices_offset = sizeof(int64_t) * (3 + n_rows + 1) + sizeof(int32_t) * csr_start;
    csr_file.seekg(indices_offset);
    csr_file.read(reinterpret_cast<char *>(local_indices.data()), sizeof(int32_t) * local_nnz);

    // Read data.
    std::vector<float> local_data(local_nnz);
    size_t data_offset = sizeof(int64_t) * (3 + n_rows + 1) + sizeof(int32_t) * nnz + sizeof(float) * csr_start;
    csr_file.seekg(data_offset);
    csr_file.read(reinterpret_cast<char *>(local_data.data()), sizeof(float) * local_nnz);
    csr_file.close();

    return new CSRMatrix(local_n_rows, n_cols, local_nnz, n_rows, nnz, local_indptr.data(), local_indices.data(), local_data.data());
}

static void get_gt(const std::string gt_path, uint32_t *&I, uint32_t &n, uint32_t &d)
{
    std::ifstream infile(gt_path, std::ios::binary);

    if (infile.fail())
    {
        std::cerr << std::string("Failed to open file ") + gt_path;
        exit(1);
    }
    infile.read((char *)&n, sizeof(uint32_t));
    infile.read((char *)&d, sizeof(uint32_t));
    I = new uint32_t[n * d];
    infile.read((char *)I, n * d * sizeof(uint32_t));
    infile.close();
}

void write_to_output(const std::string &filepath, int input_size, int world_size, int sample_size, int m, int branching_factor,
                     float index_time, float search_time, double recall, double activation_rate)
{

    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filepath << " for writing.\n";
        return;
    }

    file << input_size << ","
         << world_size << ","
         << sample_size << ","
         << m << ","
         << branching_factor << ","
         << index_time << ","
         << search_time << ","
         << recall << ","
         << activation_rate << "\n";

    file.close();
}

CSRMatrix *sample_input(CSRMatrix *datamatrix, int num_samples, int global_num_samples, MPI_Comm comm)
{

    if (num_samples > datamatrix->nrow)
    {
        throw std::invalid_argument("Local number of samples exceeds number of rows in the local matrix.");
    }

    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 rng(rd());

    // Create a vector of indices and shuffle.
    std::vector<int> indices(datamatrix->nrow);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    int dim = datamatrix->ncol;

    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    // Allocate memory safely.
    std::vector<int64_t> sampled_indptr(num_samples);
    if (rank == world_size - 1)
    {
        // Last rank needs one more entry for the end of the last row.
        sampled_indptr.resize(num_samples + 1);
    }

    std::vector<int32_t> sampled_indices;
    std::vector<float> sampled_data;

    int current_indptr = 0;

    // Fill sampled_data.
    for (int i = 0; i < num_samples; ++i)
    {
        int idx = indices[i];
        int start = datamatrix->indptr[idx];
        int end = datamatrix->indptr[idx + 1];

        int row_size = end - start;

        sampled_indices.reserve(sampled_indices.size() + row_size);
        sampled_data.reserve(sampled_data.size() + row_size);

        // Copy the indices and data for the sampled row.
        for (int j = start; j < end; ++j)
        {
            sampled_indices.push_back(datamatrix->indices_data[j].indice);
            sampled_data.push_back(datamatrix->indices_data[j].data);
        }

        sampled_indptr[i] = current_indptr;
        current_indptr += row_size;
    }

    if (rank == world_size - 1)
    {
        // Last rank needs to set the last indptr entry to the current indptr.
        sampled_indptr[num_samples] = current_indptr;
    }

    // Since all the indptr entries are local, we need to adjust them to be global.
    int local_sampled_nnz = sampled_indices.size();
    int offset = 0;
    MPI_Exscan(&local_sampled_nnz, &offset, 1, MPI_INT, MPI_SUM, comm);

    // Apply the offset to the local indptrs.
    for (int i = 0; i < sampled_indptr.size(); ++i)
    {
        sampled_indptr[i] += offset;
    }

    // Gather all local indptr sizes.
    std::vector<int> indptr_sizes(world_size);
    int local_indptr_size = sampled_indptr.size();
    MPI_Allgather(&local_indptr_size, 1, MPI_INT, indptr_sizes.data(), 1, MPI_INT, comm);

    // Calculate displacements to gather indptrs.
    std::vector<int> indptr_displs(world_size, 0);
    for (int i = 1; i < world_size; ++i)
    {
        indptr_displs[i] = indptr_displs[i - 1] + indptr_sizes[i - 1];
    }

    // Gather all local indptrs.
    std::vector<int64_t> global_indptr(global_num_samples + 1);
    MPI_Allgatherv(sampled_indptr.data(), local_indptr_size, MPI_INT64_T,
                   global_indptr.data(), indptr_sizes.data(), indptr_displs.data(), MPI_INT64_T, comm);

    // Gather recv counnts for indices and data.
    std::vector<int> recv_indices_counts(world_size);
    MPI_Allgather(&local_sampled_nnz, 1, MPI_INT,
                  recv_indices_counts.data(), 1, MPI_INT, comm);

    // Calculate total number of indices and data to be gathered.
    int total_sampled_nnz = std::accumulate(recv_indices_counts.begin(), recv_indices_counts.end(), 0);

    std::vector<int32_t> global_indices(total_sampled_nnz);
    std::vector<float> global_data(total_sampled_nnz);

    // Calculate displacements for indices and data.
    std::vector<int> recv_displs(world_size, 0);
    for (int i = 1; i < world_size; ++i)
    {
        recv_displs[i] = recv_displs[i - 1] + recv_indices_counts[i - 1];
    }

    // Gather indices and data from all processes.
    MPI_Allgatherv(sampled_indices.data(), num_samples + 1, MPI_INT32_T,
                   global_indices.data(), recv_indices_counts.data(), recv_displs.data(), MPI_INT32_T, comm);

    MPI_Allgatherv(sampled_data.data(), sampled_data.size(), MPI_FLOAT,
                   global_data.data(), recv_indices_counts.data(), recv_displs.data(), MPI_FLOAT, comm);

    return new CSRMatrix(
        global_num_samples, datamatrix->ncol, total_sampled_nnz, global_num_samples, total_sampled_nnz,
        global_indptr.data(), global_indices.data(), global_data.data());
}

void greedy_grouping(CSRMatrix *sampled_matrix, int w, int sample_size, int dim, sparse_hnswlib::HierarchicalNSW<float> &meta_hnsw,
                     int k_neighbors, std::vector<int> &sample_to_group)
{

    sample_to_group.assign(sample_size, -1);

    std::vector<int> group_sizes(w, 0);
    int max_group_size = (sample_size + w - 1) / w;

    // Initializes the first w samples, each to a unique group.
    for (int i = 0; i < std::min(w, sample_size); ++i)
    {
        sample_to_group[i] = i;
        group_sizes[i]++;
    }

    // Assign remaining samples to groups based on their neighbors.
    for (int i = w; i < sample_size; ++i)
    {
        std::vector<int> scores(w, 0);
        std::priority_queue<std::pair<float, sparse_hnswlib::labeltype>> neighbors = meta_hnsw.searchKnn(i, k_neighbors, sampled_matrix);
        while (!neighbors.empty())
        {
            int neighbor = neighbors.top().second;
            neighbors.pop();
            if (neighbor == i)
            {
                continue;
            }

            int group = sample_to_group[neighbor];
            if (group != -1 && group_sizes[group] < max_group_size)
                scores[group]++;
        }

        std::priority_queue<std::pair<int, int>> best_groups;
        for (int group = 0; group < w; ++group)
        {
            if (group_sizes[group] < max_group_size)
            {
                best_groups.push({scores[group], group});
            }
        }

        bool best_group_found = false;
        while (!best_groups.empty())
        {
            int best_group = best_groups.top().second;
            best_groups.pop();

            if (group_sizes[best_group] < max_group_size)
            {
                sample_to_group[i] = best_group;
                group_sizes[best_group]++;
                best_group_found = true;
            }

            if (best_group_found)
                break;
        }

        if (!best_group_found)
        {
            int best_group = std::min_element(group_sizes.begin(), group_sizes.end()) - group_sizes.begin();
            sample_to_group[i] = best_group;
            group_sizes[best_group]++;
        }
    }

    // std::cout << "Print group sizes:\n";
    // for (int i = 0; i < w; ++i)
    // {
    //     std::cout << "Group " << i << ": " << group_sizes[i] << " samples.\n";
    //     std::cout << "samples of group " << i << ": ";
    //     for (int j = 0; j < sample_size; ++j)
    //     {
    //         if (sample_to_group[j] == i)
    //         {
    //             std::cout << j << " ";
    //         }
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "Greedy grouping completed.\n";
}

double distribute_data_matrix(CSRMatrix *datamatrix, CSRMatrix **local_datamatrix, std::vector<int> *recv_label_buffer, sparse_hnswlib::HierarchicalNSW<float> &meta_hnsw,
                              std::vector<int> &sample_to_group, int k, int input_size, int dim, int rank, int world_size)
{

    int label_offset = rank * (input_size / world_size);
    std::vector<std::vector<int64_t>> labels_to_send(world_size);
    std::vector<std::vector<int64_t>> indptr_to_send(world_size);
    std::vector<std::vector<int32_t>> indices_to_send(world_size);
    std::vector<std::vector<float>> data_to_send(world_size);

    std::vector<int64_t> current_indptr(world_size, 0);

    double activations = 0.0;
    for (int i = 0; i < datamatrix->nrow; ++i)
    {
        std::priority_queue<std::pair<float, sparse_hnswlib::labeltype>> samples = meta_hnsw.searchKnn(i, k, datamatrix);

        std::unordered_set<int> visited_groups;
        int label = label_offset + i;

        int start = datamatrix->indptr[i];
        int end = datamatrix->indptr[i + 1];
        while (!samples.empty())
        {
            int sample = samples.top().second;
            samples.pop();
            int group = sample_to_group[sample];

            if (visited_groups.find(group) != visited_groups.end())
            {
                continue;
            }

            labels_to_send[group].push_back(label);
            indptr_to_send[group].push_back(current_indptr[group]);
            for (int j = start; j < end; ++j)
            {
                indices_to_send[group].push_back(datamatrix->indices_data[j].indice);
                data_to_send[group].push_back(datamatrix->indices_data[j].data);
            }

            visited_groups.insert(group);
            activations++;

            current_indptr[group] += (end - start);
        }
    }

    std::vector<int> send_label_buffer;
    std::vector<int64_t> send_indptr_buffer;
    std::vector<int> send_label_counts(world_size), recv_label_counts(world_size);
    // Flatten the labels_to_send and indptr_to_send.
    for (int i = 0; i < world_size; ++i)
    {
        send_label_counts[i] = labels_to_send[i].size();
        send_label_buffer.insert(send_label_buffer.end(), labels_to_send[i].begin(), labels_to_send[i].end());
        send_indptr_buffer.insert(send_indptr_buffer.end(), indptr_to_send[i].begin(), indptr_to_send[i].end());
    }

    // Gather the sizes of the data to be received from each process.
    MPI_Alltoall(send_label_counts.data(), 1, MPI_INT, recv_label_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate the displacements (offsets) for send and receive buffers for each processor.
    std::vector<int> send_label_offsets(world_size, 0), recv_label_offsets(world_size, 0);
    for (int i = 1; i < world_size; ++i)
    {
        send_label_offsets[i] = send_label_offsets[i - 1] + send_label_counts[i - 1];
        recv_label_offsets[i] = recv_label_offsets[i - 1] + recv_label_counts[i - 1];
    }

    int total_recv_label_count = recv_label_offsets.back() + recv_label_counts.back();
    recv_label_buffer->resize(total_recv_label_count);
    std::vector<int64_t> recv_indptr_buffer(total_recv_label_count);

    MPI_Alltoallv(send_label_buffer.data(), send_label_counts.data(), send_label_offsets.data(), MPI_INT,
                  recv_label_buffer->data(), recv_label_counts.data(), recv_label_offsets.data(), MPI_INT,
                  MPI_COMM_WORLD);

    MPI_Alltoallv(send_indptr_buffer.data(), send_label_counts.data(), send_label_offsets.data(), MPI_INT64_T,
                  recv_indptr_buffer.data(), recv_label_counts.data(), recv_label_offsets.data(), MPI_INT64_T,
                  MPI_COMM_WORLD);

    std::vector<int32_t> send_indices_buffer;
    std::vector<float> send_data_buffer;
    std::vector<int> send_data_counts(world_size), recv_data_counts(world_size);
    // Flatten the indices_to_send and data_to_send.
    for (int i = 0; i < world_size; ++i)
    {
        send_data_counts[i] = data_to_send[i].size();
        send_indices_buffer.insert(send_indices_buffer.end(), indices_to_send[i].begin(), indices_to_send[i].end());
        send_data_buffer.insert(send_data_buffer.end(), data_to_send[i].begin(), data_to_send[i].end());
    }

    // Gather the sizes of the data to be received from each process.
    MPI_Alltoall(send_data_counts.data(), 1, MPI_INT, recv_data_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate the displacements (offsets) for send and receive buffers for each processor.
    std::vector<int> send_data_offsets(world_size, 0), recv_data_offsets(world_size, 0);
    for (int i = 1; i < world_size; ++i)
    {
        send_data_offsets[i] = send_data_offsets[i - 1] + send_data_counts[i - 1];
        recv_data_offsets[i] = recv_data_offsets[i - 1] + recv_data_counts[i - 1];
    }

    int total_recv_data_count = recv_data_offsets.back() + recv_data_counts.back();
    std::vector<int32_t> recv_indices_buffer(total_recv_data_count);
    std::vector<float> recv_data_buffer(total_recv_data_count);

    MPI_Alltoallv(send_indices_buffer.data(), send_data_counts.data(), send_data_offsets.data(), MPI_INT32_T,
                  recv_indices_buffer.data(), recv_data_counts.data(), recv_data_offsets.data(), MPI_INT32_T,
                  MPI_COMM_WORLD);

    MPI_Alltoallv(send_data_buffer.data(), send_data_counts.data(), send_data_offsets.data(), MPI_FLOAT,
                  recv_data_buffer.data(), recv_data_counts.data(), recv_data_offsets.data(), MPI_FLOAT,
                  MPI_COMM_WORLD);

    // Calculate final indptr for the local datamatrix.
    std::vector<int64_t> final_indptr(total_recv_label_count + 1, 0);
    int indptr_index_offset = 0;
    int dataptr_offset = 0;
    int recv_count_index = 0;

    for (int i = 0; i < recv_indptr_buffer.size(); ++i)
    {
        if (i >= indptr_index_offset + recv_label_counts[recv_count_index])
        {
            indptr_index_offset += recv_label_counts[recv_count_index];
            dataptr_offset += recv_data_counts[recv_count_index];
            recv_count_index++;
        }

        final_indptr[i] = recv_indptr_buffer[i] + dataptr_offset;
    }

    dataptr_offset += recv_data_counts[recv_count_index];
    final_indptr[total_recv_label_count] = recv_data_buffer.size();

    // Build final local datamatrix.
    *local_datamatrix = new CSRMatrix(total_recv_label_count, dim, total_recv_data_count, datamatrix->global_nrow, datamatrix->global_nnz,
                                      final_indptr.data(), recv_indices_buffer.data(), recv_data_buffer.data());
    return activations;
}

int get_owner(int label, int input_size, int world_size)
{

    return std::floor(label * world_size / input_size);
}

void distribute_local_results(std::vector<std::vector<int>> &query_labels_to_send,
                              std::vector<std::vector<int>> &result_ids_to_send,
                              std::vector<std::vector<float>> &result_dists_to_send,
                              std::vector<int> *result_recv_labels,
                              std::vector<int> *result_recv_ids,
                              std::vector<float> *result_recv_dists)
{

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> query_label_send_counts(world_size), query_label_recv_counts(world_size);
    std::vector<int> query_label_send_offsets(world_size), query_label_recv_offsets(world_size);

    std::vector<int> result_send_counts(world_size), result_recv_counts(world_size);
    std::vector<int> result_send_offsets(world_size), result_recv_offsets(world_size);

    // Flatten local result buffers.
    std::vector<int> local_result_flat_ids;
    std::vector<float> local_result_flat_dists;
    std::vector<int> local_flat_query_labels;

    for (int i = 0; i < world_size; ++i)
    {
        query_label_send_counts[i] = query_labels_to_send[i].size();
        query_label_send_offsets[i] = i > 0 ? query_label_send_offsets[i - 1] + query_label_send_counts[i - 1] : 0;

        result_send_counts[i] = result_ids_to_send[i].size();
        result_send_offsets[i] = i > 0 ? result_send_offsets[i - 1] + result_send_counts[i - 1] : 0;

        local_result_flat_ids.insert(local_result_flat_ids.end(), result_ids_to_send[i].begin(), result_ids_to_send[i].end());
        local_result_flat_dists.insert(local_result_flat_dists.end(), result_dists_to_send[i].begin(), result_dists_to_send[i].end());
        local_flat_query_labels.insert(local_flat_query_labels.end(), query_labels_to_send[i].begin(), query_labels_to_send[i].end());
    }

    std::cout << "Rank: " << rank << " Sending " << local_result_flat_ids.size() << " ids, " << local_result_flat_dists.size() << " distances, and "
              << local_flat_query_labels.size() << " labels." << std::endl;

    // Send results back to relevant owner processors.
    MPI_Alltoall(query_label_send_counts.data(), 1, MPI_INT, query_label_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(result_send_counts.data(), 1, MPI_INT, result_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 1; i < world_size; ++i)
    {
        query_label_recv_offsets[i] = query_label_recv_offsets[i - 1] + query_label_recv_counts[i - 1];
        result_recv_offsets[i] = result_recv_offsets[i - 1] + result_recv_counts[i - 1];
    }

    result_recv_labels->resize(std::accumulate(query_label_recv_counts.begin(), query_label_recv_counts.end(), 0));
    result_recv_ids->resize(std::accumulate(result_recv_counts.begin(), result_recv_counts.end(), 0));
    result_recv_dists->resize(result_recv_ids->size());

    MPI_Alltoallv(local_flat_query_labels.data(), query_label_send_counts.data(), query_label_send_offsets.data(), MPI_INT,
                  result_recv_labels->data(), query_label_recv_counts.data(), query_label_recv_offsets.data(), MPI_INT,
                  MPI_COMM_WORLD);

    std::cout << "Rank: " << rank << " Received " << result_recv_labels->size() << " query labels." << std::endl;

    MPI_Alltoallv(local_result_flat_ids.data(), result_send_counts.data(), result_send_offsets.data(), MPI_INT,
                  result_recv_ids->data(), result_recv_counts.data(), result_recv_offsets.data(), MPI_INT, MPI_COMM_WORLD);

    MPI_Alltoallv(local_result_flat_dists.data(), result_send_counts.data(), result_send_offsets.data(), MPI_FLOAT,
                  result_recv_dists->data(), result_recv_counts.data(), result_recv_offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);

    std::cout << "Rank: " << rank << " Received " << result_recv_ids->size() << " ids, " << result_recv_dists->size() << " distances, and "
              << result_recv_labels->size() << " labels." << std::endl;
}

void calculate_final_results(std::unordered_map<int, std::priority_queue<std::pair<float, int>>> *query_results, 
                             std::vector<int> &result_recv_labels,
                             std::vector<int> &result_recv_ids,
                             std::vector<float> &result_recv_dists, int d) {

    for (int i = 0; i < result_recv_labels.size(); ++i)
    {
        int query_label = result_recv_labels[i];
        for (int j = 0; j < d; ++j)
        {
            int idx = i * d + j;
            (*query_results)[query_label].emplace(result_recv_dists[idx], result_recv_ids[idx]);

            auto &pq = (*query_results)[query_label];
            while (pq.size() > d)
                pq.pop();
        }
    }
}

void distribute_final_results_to_root(std::unordered_map<int, std::priority_queue<std::pair<float, int>>> &query_results, 
    std::vector<int> *gathered_labels, std::vector<int> *gathered_ids,
    std::vector<float> *gathered_dists) {

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Flatten the final results for gathering.
    std::vector<int> flat_final_query_labels, flat_final_result_ids;
    std::vector<float> flat_final_result_dists;

    for (const auto &[label, pq] : query_results)
    {
        std::priority_queue<std::pair<float, int>> copy_pq = pq;
        std::vector<std::pair<float, int>> sorted_pq;

        while (!copy_pq.empty())
        {
            sorted_pq.push_back(copy_pq.top());
            copy_pq.pop();
        }
        std::reverse(sorted_pq.begin(), sorted_pq.end());

        flat_final_query_labels.push_back(label);
        for (auto &[dist, id] : sorted_pq)
        {
            flat_final_result_ids.push_back(id);
            flat_final_result_dists.push_back(dist);
        }
    }

    std::cout << "Rank: " << rank << " Local results prepared for " << flat_final_query_labels.size() << " queries." << std::endl;

    int final_query_label_send_count = flat_final_query_labels.size();
    std::vector<int> final_query_label_recv_counts(world_size), final_query_label_displs(world_size);

    int final_result_send_count = flat_final_result_ids.size();
    std::vector<int> final_result_recv_counts(world_size), final_result_displs(world_size);

    MPI_Gather(&final_query_label_send_count, 1, MPI_INT, final_query_label_recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&final_result_send_count, 1, MPI_INT, final_result_recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        final_query_label_displs[0] = final_result_displs[0] = 0;
        for (int i = 1; i < world_size; ++i)
        {
            final_query_label_displs[i] = final_query_label_displs[i - 1] + final_query_label_recv_counts[i - 1];
            final_result_displs[i] = final_result_displs[i - 1] + final_result_recv_counts[i - 1];
        }

        int total_labels = final_query_label_displs.back() + final_query_label_recv_counts.back();
        int total_results = final_result_displs.back() + final_result_recv_counts.back();

        gathered_labels->resize(total_labels);
        gathered_ids->resize(total_results);
        gathered_dists->resize(total_results);
    }

    // Gather query labels.
    MPI_Gatherv(flat_final_query_labels.data(), final_query_label_send_count, MPI_INT,
                gathered_labels->data(), final_query_label_recv_counts.data(), final_query_label_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Gather final result ids.
    MPI_Gatherv(flat_final_result_ids.data(), final_result_send_count, MPI_INT,
                gathered_ids->data(), final_result_recv_counts.data(), final_result_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Gather final result distances.
    MPI_Gatherv(flat_final_result_dists.data(), final_result_send_count, MPI_FLOAT,
                gathered_dists->data(), final_result_recv_counts.data(), final_result_displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    std::cout << "Rank: " << rank << " Gathered results from all processes." << std::endl;
}

double calculate_recall(const std::vector<int> &gathered_labels, const std::vector<int> &gathered_ids, uint32_t *&I, int d)
{

    int total_hits = 0;
    for (int i = 0; i < gathered_labels.size(); ++i)
    {
        int query_label = gathered_labels[i];
        std::vector<int> pred_ids = std::vector<int>(gathered_ids.begin() + i * d, gathered_ids.begin() + (i + 1) * d);
        std::unordered_set<uint32_t> gt_neighbors(I + query_label * d, I + (query_label + 1) * d);

        int hit_count = 0;
        for (int pid : pred_ids)
        {
            if (gt_neighbors.count(pid))
                hit_count++;
        }

        total_hits += hit_count;
    }

    return static_cast<double>(total_hits) / (gathered_labels.size() * d);
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 9)
    {
        std::cerr << "Usage: " << argv[0] << " <input_filepath> <global_sample_size> <m> <branching_factor> <M> <ef_construction> <query_filepath> <gt_filepath>" << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
    std::string input_filepath = argv[1];
    int global_sample_size = std::stoi(argv[2]);
    int m = std::stoi(argv[3]);
    int k = std::stoi(argv[4]);
    int M = std::stoi(argv[5]);
    int ef_construction = std::stoi(argv[6]);
    std::string query_filepath = argv[7];
    std::string gt_filepath = argv[8];

    CSRMatrix *datamatrix = read_csr(input_filepath, rank, world_size);

    int input_size = datamatrix->nrow;
    int dim = datamatrix->ncol;
    int max_elements = datamatrix->nrow;

    MPI_Barrier(MPI_COMM_WORLD);
    double hnsw_build_start = MPI_Wtime();

    int sample_size = global_sample_size / world_size;
    CSRMatrix *sample_matrix = sample_input(datamatrix, sample_size, global_sample_size, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Data sampling completed. Sampled " << sample_matrix->nrow << " rows with nnz: "
                  << sample_matrix->nnz << " from the input data matrix." << std::endl;
    }

    sparse_hnswlib::InnerProductSpace space(dim);
    sparse_hnswlib::HierarchicalNSW<float> *meta_hnsw =
        new sparse_hnswlib::HierarchicalNSW<float>(&space, datamatrix, global_sample_size, M, ef_construction);
    for (int i = 0; i < global_sample_size; i++)
    {
        meta_hnsw->addPoint(i, i);
    }

    if (rank == 0)
    {
        std::cout << "Meta HNSW index built with " << global_sample_size << " samples." << std::endl;
    }

    std::vector<int> sample_to_group(global_sample_size);

    if (rank == 0)
    {
        greedy_grouping(sample_matrix, world_size, global_sample_size, dim, *meta_hnsw, k, sample_to_group);
    }
    MPI_Bcast(sample_to_group.data(), global_sample_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Greedy grouping completed.\n";
    }

    std::vector<int> local_labels;
    CSRMatrix *local_datamatrix;
    double _ = distribute_data_matrix(datamatrix, &local_datamatrix, &local_labels, *meta_hnsw, sample_to_group, k, input_size, dim, rank, world_size);

    int local_input_size = local_datamatrix->nrow;

    if (rank == 0)
    {
        std::cout << "Local data distribution completed. Local input size: " << local_input_size << std::endl;
    }

    // Initiate hnsw index.
    sparse_hnswlib::HierarchicalNSW<float> *local_hnsw =
        new sparse_hnswlib::HierarchicalNSW<float>(&space, local_datamatrix, local_input_size, M, ef_construction);

    if (rank == 0)
    {
        std::cout << "Local HNSW index initialized." << std::endl;
    }

    // Add data to hnsw index.
    for (int i = 0; i < local_input_size; i++)
    {
        local_hnsw->addPoint(i, i);
    }

    std::cout << "Rank: " << rank << " Local HNSW index built with " << local_input_size << " data points." << std::endl;

    double hnsw_build_end = MPI_Wtime();
    double local_hnsw_build_duration = hnsw_build_end - hnsw_build_start;
    double global_hnsw_build_duration;
    MPI_Reduce(&local_hnsw_build_duration, &global_hnsw_build_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "Time taken to build HNSW index: " << global_hnsw_build_duration << " seconds" << std::endl;
    }

    CSRMatrix *query_datamatrix = read_csr(query_filepath, rank, world_size);

    uint32_t *I = nullptr;
    uint32_t n, d;
    if (rank == 0)
    {
        get_gt(gt_filepath, I, n, d);
    }

    MPI_Bcast(&n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double search_start = MPI_Wtime();

    int query_input_size = query_datamatrix->global_nrow;

    std::vector<int> local_query_labels;
    CSRMatrix *local_query_datamatrix;
    double activations = distribute_data_matrix(query_datamatrix, &local_query_datamatrix, &local_query_labels, *meta_hnsw, sample_to_group, k, query_input_size, dim, rank, world_size);

    int local_query_input_size = local_query_datamatrix->nrow;

    std::vector<std::vector<int>> query_labels_to_send(world_size);
    std::vector<std::vector<int>> result_ids_to_send(world_size);
    std::vector<std::vector<float>> result_dists_to_send(world_size);

    // Process all received queries.
    if (local_query_input_size > 0)
    {
        // Find nearest neighbors of the queries using HNSW.
        for (int i = 0; i < local_query_input_size; ++i)
        {
            std::priority_queue<std::pair<float, sparse_hnswlib::labeltype>> results = local_hnsw->searchKnn(i, d, local_query_datamatrix);

            int query_label = local_query_labels[i];
            int owner = get_owner(query_label, query_input_size, world_size);

            query_labels_to_send[owner].push_back(query_label);

            int result_index = 0;
            while (!results.empty())
            {
                float distance = results.top().first;
                int label = local_labels[results.top().second];
                results.pop();

                result_ids_to_send[owner].push_back(label);
                result_dists_to_send[owner].push_back(distance);
            }
        }
    }

    std::cout << "Rank: " << rank << " Local search completed for " << local_query_input_size << " queries." << std::endl;

    std::vector<int> result_recv_labels;
    std::vector<int> result_recv_ids;
    std::vector<float> result_recv_dists;

    distribute_local_results(query_labels_to_send, result_ids_to_send, result_dists_to_send,
                             &result_recv_labels, &result_recv_ids, &result_recv_dists);

    // Calculate final top-d results for queries assigned to the processor.
    std::unordered_map<int, std::priority_queue<std::pair<float, int>>> query_results;
    calculate_final_results(&query_results, result_recv_labels, result_recv_ids, result_recv_dists, d);

    std::cout << "Rank: " << rank << " Search completed for " << local_query_input_size << " queries." << std::endl;

    double search_end = MPI_Wtime();
    double local_search_duration = search_end - search_start;
    double global_search_duration;
    MPI_Reduce(&local_search_duration, &global_search_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double global_activations;
    MPI_Reduce(&activations, &global_activations, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<int> gathered_labels, gathered_ids;
    std::vector<float> gathered_dists;
    distribute_final_results_to_root(query_results, &gathered_labels, &gathered_ids, &gathered_dists);

    if (rank == 0)
    {
        double global_activation_rate = global_activations / (query_input_size * world_size);
        std::cout << "Activation rate: " << global_activation_rate << std::endl;
        std::cout << "Time taken for search: " << global_search_duration << " seconds\n";

        double recall = calculate_recall(gathered_labels, gathered_ids, I, d);
        std::cout << "Recall@" << d << " = " << recall << std::endl;
    }

    MPI_Finalize();

    return 0;
}
