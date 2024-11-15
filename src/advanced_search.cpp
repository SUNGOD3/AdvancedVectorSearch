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

    // Build KD-tree
    std::vector<size_t> indices(m_num_vectors);
    for (size_t i = 0; i < m_num_vectors; ++i) {
        indices[i] = i;
    }
    build_tree(root, indices, 0);
}

AdvancedKNNSearch::~AdvancedKNNSearch() {
    delete[] m_data;
}

void AdvancedKNNSearch::build_tree(std::unique_ptr<Node>& node, std::vector<size_t>& indices, int depth) {
    if (indices.empty()) {
        return;
    }

    node.reset(new Node());
    
    if (indices.size() == 1) {
        node->idx = indices[0];
        node->pivot = m_data + indices[0] * m_vector_size;
        return;
    }

    // Choose splitting dimension - use the dimension with highest variance
    size_t best_dim = 0;
    float max_variance = -1;
    
    for (size_t d = 0; d < m_vector_size; d++) {
        float mean = 0, variance = 0;
        
        // Calculate mean
        for (size_t i = 0; i < indices.size(); i++) {
            mean += m_data[indices[i] * m_vector_size + d];
        }
        mean /= indices.size();
        
        // Calculate variance
        for (size_t i = 0; i < indices.size(); i++) {
            float diff = m_data[indices[i] * m_vector_size + d] - mean;
            variance += diff * diff;
        }
        
        if (variance > max_variance) {
            max_variance = variance;
            best_dim = d;
        }
    }

    size_t mid = indices.size() / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
        [this, best_dim](size_t a, size_t b) {
            return m_data[a * m_vector_size + best_dim] < m_data[b * m_vector_size + best_dim];
        });

    node->idx = indices[mid];
    node->pivot = m_data + indices[mid] * m_vector_size;
    node->split_dim = best_dim;

    std::vector<size_t> left_indices(indices.begin(), indices.begin() + mid);
    std::vector<size_t> right_indices(indices.begin() + mid + 1, indices.end());

    build_tree(node->left, left_indices, depth + 1);
    build_tree(node->right, right_indices, depth + 1);
}

void AdvancedKNNSearch::search_tree(const Node* node, const float* query,
                                  std::priority_queue<std::pair<float, size_t>>& results,
                                  float& worst_dist, int k) const {
    if (!node) return;

    float dist = cosine_distance(query, node->pivot, m_vector_size);
    
    if (results.size() < static_cast<size_t>(k)) {
        results.push({-dist, node->idx});  // Negative dist for max-heap to work as min-heap
        if (results.size() == static_cast<size_t>(k)) {
            worst_dist = -results.top().first;
        }
    } else if (dist < worst_dist) {
        results.pop();
        results.push({-dist, node->idx});
        worst_dist = -results.top().first;
    }

    if (!node->left && !node->right) return;

    float diff = query[node->split_dim] - node->pivot[node->split_dim];
    const Node* first = diff <= 0 ? node->left.get() : node->right.get();
    const Node* second = diff <= 0 ? node->right.get() : node->left.get();

    // Always search the closer branch first
    search_tree(first, query, results, worst_dist, k);
    
    // Only search the other branch if it could contain better points
    float split_dist = std::abs(diff);
    if (results.size() < static_cast<size_t>(k) || split_dist < worst_dist) {
        search_tree(second, query, results, worst_dist, k);
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
    
    // Use priority queue for maintaining top-k results
    std::priority_queue<std::pair<float, size_t>> results;
    float worst_dist = std::numeric_limits<float>::max();
    
    search_tree(root.get(), query_ptr, results, worst_dist, k);
    
    // Convert results to sorted array
    std::vector<std::pair<float, size_t>> sorted_results;
    while (!results.empty()) {
        auto result = results.top();
        sorted_results.push_back({-result.first, result.second});  // Convert back to positive distances
        results.pop();
    }
    std::reverse(sorted_results.begin(), sorted_results.end());  // Reverse to get ascending order
    
    // Create return array
    py::array_t<int> result(sorted_results.size());
    for (size_t i = 0; i < sorted_results.size(); i++) {
        result.mutable_at(i) = sorted_results[i].second;
    }

    return result;
}