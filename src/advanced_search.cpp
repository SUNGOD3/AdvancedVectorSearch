// src/advanced_search.cpp
#include "advanced_search.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>

float BaseAdvancedSearch::cosine_distance(const float* a, const float* b, size_t size) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    
    #pragma omp simd reduction(+:dot,denom_a,denom_b)
    for (size_t i = 0; i < size; ++i) {
        dot += a[i] * b[i];
        denom_a += a[i] * a[i];
        denom_b += b[i] * b[i];
    }

    return 1.0f - dot / std::sqrt(denom_a * denom_b);
}

// Advanced Linear Search Implementation
AdvancedLinearSearch::AdvancedLinearSearch(py::array_t<float> vectors) {
    py::buffer_info buf = vectors.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    m_num_vectors = buf.shape[0];
    m_vector_size = buf.shape[1];

    size_t total_size = m_num_vectors * m_vector_size;
    m_data = new float[total_size];
    
    std::memcpy(m_data, buf.ptr, sizeof(float) * total_size);
}

AdvancedLinearSearch::~AdvancedLinearSearch() {
    delete[] m_data;
}

void AdvancedLinearSearch::parallel_sort(std::pair<float, size_t>* distances, int k) {
    int num_threads = 4;
    int block_size = (k + num_threads - 1) / num_threads;
    
    #pragma omp parallel for 
    for (int i = 0; i < k; i += block_size) {
        int block_end = std::min(i + block_size, k);
        std::sort(distances + i, distances + block_end);
    }
    
    for (int merge_size = block_size; merge_size < k; merge_size *= 2) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < k; i += 2 * merge_size) {
            int mid = std::min(i + merge_size, k);
            int end = std::min(i + 2 * merge_size, k);
            if (end > mid) {
                std::inplace_merge(distances + i, distances + mid, distances + end);
            }
        }
    }
}

py::array_t<int> AdvancedLinearSearch::search(py::array_t<float> query, int k) {
    py::buffer_info buf = query.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    if (static_cast<size_t>(buf.shape[0]) != m_vector_size) {
        throw std::runtime_error("Query vector dimension mismatch");
    }
    
    const float* query_ptr = static_cast<float*>(buf.ptr);
    std::pair<float, size_t>* distances = new std::pair<float, size_t>[m_num_vectors];

    #pragma omp parallel for
    for (size_t i = 0; i < m_num_vectors; ++i) {
        distances[i] = {cosine_distance(query_ptr, m_data + i * m_vector_size, m_vector_size), i};
    }

    k = std::min(k, static_cast<int>(m_num_vectors));
    std::nth_element(distances, distances + k, distances + m_num_vectors);
    parallel_sort(distances, k);

    py::array_t<int> result(k);
    #pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        result.mutable_at(i) = distances[i].second;
    }

    delete[] distances;
    return result;
}

// Advanced KNN Search Implementation
AdvancedKNNSearch::AdvancedKNNSearch(py::array_t<float> vectors) {
    py::buffer_info buf = vectors.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    m_num_vectors = buf.shape[0];
    m_vector_size = buf.shape[1];

    size_t total_size = m_num_vectors * m_vector_size;
    m_data = new float[total_size];
    std::memcpy(m_data, buf.ptr, sizeof(float) * total_size);

    // Build ball tree
    std::vector<size_t> indices(m_num_vectors);
    for (size_t i = 0; i < m_num_vectors; ++i) {
        indices[i] = i;
    }
    build_tree(root, indices);
}

AdvancedKNNSearch::~AdvancedKNNSearch() {
    delete[] m_data;
}

float AdvancedKNNSearch::compute_radius(const std::vector<size_t>& indices, size_t center_idx) {
    float max_dist = 0.0f;
    const float* center = m_data + center_idx * m_vector_size;
    
    for (size_t idx : indices) {
        if (idx == center_idx) continue;
        const float* point = m_data + idx * m_vector_size;
        float dist = cosine_distance(center, point, m_vector_size);
        max_dist = std::max(max_dist, dist);
    }
    return max_dist;
}

size_t AdvancedKNNSearch::find_furthest_point(const std::vector<size_t>& indices, size_t center_idx) {
    float max_dist = -1.0f;
    size_t furthest_idx = center_idx;
    const float* center = m_data + center_idx * m_vector_size;
    
    for (size_t idx : indices) {
        if (idx == center_idx) continue;
        const float* point = m_data + idx * m_vector_size;
        float dist = cosine_distance(center, point, m_vector_size);
        if (dist > max_dist) {
            max_dist = dist;
            furthest_idx = idx;
        }
    }
    return furthest_idx;
}

void AdvancedKNNSearch::build_tree(std::unique_ptr<BallNode>& node, std::vector<size_t>& indices) {
    if (indices.empty()) return;
    node = std::make_unique<BallNode>();
    
    if (indices.size() <= 128) {
        node->points = indices;
        node->center_idx = indices[0];
        node->radius = compute_radius(indices, node->center_idx);
        return;
    }

    const size_t sample_size = std::min(size_t(512), indices.size());
    
    std::vector<float> mean(m_vector_size, 0.0f);
    std::vector<size_t> sampled_indices;
    
    for (size_t i = 0; i < sample_size; ++i) {
        size_t rand_idx = i < indices.size() ? i : rand() % indices.size();
        sampled_indices.push_back(indices[rand_idx]);
        
        const float* point = m_data + indices[rand_idx] * m_vector_size;
        for (size_t j = 0; j < m_vector_size; ++j) {
            mean[j] += point[j];
        }
    }
    
    for (size_t j = 0; j < m_vector_size; ++j) {
        mean[j] /= sample_size;
    }
    
    float min_distance = std::numeric_limits<float>::max();
    size_t center_idx = indices[0];
    
    for (size_t idx : sampled_indices) {
        const float* point = m_data + idx * m_vector_size;
        float dist = 0.0f;
        
        for (size_t j = 0; j < m_vector_size; ++j) {
            float diff = point[j] - mean[j];
            dist += diff * diff;
        }
        
        if (dist < min_distance) {
            min_distance = dist;
            center_idx = idx;
        }
    }
    
    node->center_idx = center_idx;
    
    std::vector<std::pair<float, size_t>> distances;
    distances.reserve(indices.size());
    
    const float* center_point = m_data + center_idx * m_vector_size;
    for (size_t idx : indices) {
        if (idx == center_idx) continue;
        
        const float* point = m_data + idx * m_vector_size;
        float dist = cosine_distance(center_point, point, m_vector_size);
        distances.emplace_back(dist, idx);
    }
    
    size_t mid = distances.size() / 2;
    std::nth_element(distances.begin(), distances.begin() + mid, distances.end());
    
    std::vector<size_t> left_indices{center_idx}, right_indices;
    for (const auto& dist_pair : distances) {
        if (dist_pair.first < distances[mid].first) {
            left_indices.push_back(dist_pair.second);
        } else {
            right_indices.push_back(dist_pair.second);
        }
    }
    
    node->radius = compute_radius(indices, center_idx);
    node->points = std::move(indices);
    
    if (!left_indices.empty()) build_tree(node->left, left_indices);
    if (!right_indices.empty()) build_tree(node->right, right_indices);
}

void AdvancedKNNSearch::search_ball_tree(const BallNode* node,
                                        const float* query,
                                        std::priority_queue<std::pair<float, size_t>>& results,
                                        float& worst_dist,
                                        int k) const {
    if (!node) return;

    // Calculate distance to center
    const float* center = m_data + node->center_idx * m_vector_size;
    float dist_to_center = cosine_distance(center, query, m_vector_size);
    
    // Early pruning: skip if this node cannot contain better points
    if (results.size() == static_cast<size_t>(k) && 
        dist_to_center - node->radius > worst_dist) {
        return;
    }
    
    // Process leaf node
    if (!node->left && !node->right) {
        #pragma omp parallel for
        for (size_t i = 0; i < node->points.size(); ++i) {
            size_t idx = node->points[i];
            const float* point = m_data + idx * m_vector_size;
            float dist = cosine_distance(point, query, m_vector_size);

            #pragma omp critical
            {
                // Update results if we found a better point
                if (results.size() < static_cast<size_t>(k)) {
                    results.push({dist, idx});
                    if (results.size() == static_cast<size_t>(k)) {
                        worst_dist = results.top().first;
                    }
                } else if (dist < worst_dist) {
                    results.pop();
                    results.push({dist, idx});
                    worst_dist = results.top().first;
                }
            }
        }
        return;
    }
    
    // For internal nodes, recursively search children
    if (node->left && node->right) {
        // Calculate distances to both children's centers
        const float* left_center = m_data + node->left->center_idx * m_vector_size;
        const float* right_center = m_data + node->right->center_idx * m_vector_size;
        
        float dist_to_left = cosine_distance(left_center, query, m_vector_size);
        float dist_to_right = cosine_distance(right_center, query, m_vector_size);
        
        // Visit closer node first for better pruning
        if (dist_to_left < dist_to_right) {
            search_ball_tree(node->left.get(), query, results, worst_dist, k);
            search_ball_tree(node->right.get(), query, results, worst_dist, k);
        } else {
            search_ball_tree(node->right.get(), query, results, worst_dist, k);
            search_ball_tree(node->left.get(), query, results, worst_dist, k);
        }
    } else {
        // Handle cases where only one child exists
        if (node->left) search_ball_tree(node->left.get(), query, results, worst_dist, k);
        if (node->right) search_ball_tree(node->right.get(), query, results, worst_dist, k);
    }
}

py::array_t<int> AdvancedKNNSearch::search(py::array_t<float> query, int k) {
    py::buffer_info buf = query.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    if (static_cast<size_t>(buf.shape[0]) != m_vector_size) {
        throw std::runtime_error("Query vector dimension mismatch");
    }

    const float* query_ptr = static_cast<float*>(buf.ptr);
    
    // If k equals total number of vectors, return all indices sorted by distance
    if (k >= static_cast<int>(m_num_vectors)) {
        std::vector<std::pair<float, size_t>> all_distances(m_num_vectors);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m_num_vectors; ++i) {
            float dist = cosine_distance(query_ptr, m_data + i * m_vector_size, m_vector_size);
            all_distances[i] = {dist, i};
        }
        
        std::sort(all_distances.begin(), all_distances.end());
        
        py::array_t<int> result(m_num_vectors);
        for (size_t i = 0; i < m_num_vectors; i++) {
            result.mutable_at(i) = all_distances[i].second;
        }
        return result;
    }
    
    // Regular k-NN search for k < m_num_vectors
    std::priority_queue<std::pair<float, size_t>> results;
    float worst_dist = std::numeric_limits<float>::max();
    
    search_ball_tree(root.get(), query_ptr, results, worst_dist, k);
    
    // Convert results to sorted array
    std::vector<std::pair<float, size_t>> sorted_results;
    while (!results.empty()) {
        sorted_results.push_back(results.top());
        results.pop();
    }
    std::reverse(sorted_results.begin(), sorted_results.end());
    
    // Create return array
    py::array_t<int> result(sorted_results.size());
    for (size_t i = 0; i < sorted_results.size(); i++) {
        result.mutable_at(i) = sorted_results[i].second;
    }

    return result;
}