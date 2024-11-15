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
    if (indices.empty()) {
        return;
    }

    node = std::make_unique<BallNode>();
    
    // For small number of points, store them directly
    if (indices.size() <= 10) {
        node->points = indices;
        node->center_idx = indices[0];
        node->radius = compute_radius(indices, node->center_idx);
        return;
    }

    // Fast center selection using sampling
    const size_t max_samples = 100;
    const size_t num_candidates = 5; 
    
    size_t stride = std::max(size_t(1), indices.size() / max_samples);
    std::vector<size_t> samples;
    samples.reserve(max_samples);
    
    // Systematic sampling
    for (size_t i = 0; i < indices.size(); i += stride) {
        samples.push_back(indices[i]);
        if (samples.size() >= max_samples) break;
    }
    
    std::vector<size_t> candidates;
    candidates.reserve(num_candidates);
    
    size_t first_idx = samples[rand() % samples.size()];
    candidates.push_back(first_idx);
    
    for (size_t c = 1; c < num_candidates; ++c) {
        float max_min_dist = -1.0f;
        size_t best_idx = first_idx;
        
        for (size_t sample_idx : samples) {
            float min_dist = std::numeric_limits<float>::max();
            const float* sample_ptr = m_data + sample_idx * m_vector_size;
            
            for (size_t cand_idx : candidates) {
                const float* cand_ptr = m_data + cand_idx * m_vector_size;
                float dist = cosine_distance(sample_ptr, cand_ptr, m_vector_size);
                min_dist = std::min(min_dist, dist);
            }
            
            if (min_dist > max_min_dist) {
                max_min_dist = min_dist;
                best_idx = sample_idx;
            }
        }
        
        candidates.push_back(best_idx);
    }
    
    float min_total_dist = std::numeric_limits<float>::max();
    node->center_idx = candidates[0];
    
    #pragma omp parallel for reduction(min:min_total_dist)
    for (size_t i = 0; i < candidates.size(); ++i) {
        const float* cand_ptr = m_data + candidates[i] * m_vector_size;
        float total_dist = 0.0f;
        
        for (size_t sample_idx : samples) {
            const float* sample_ptr = m_data + sample_idx * m_vector_size;
            total_dist += cosine_distance(cand_ptr, sample_ptr, m_vector_size);
        }
        
        if (total_dist < min_total_dist) {
            #pragma omp critical
            {
                if (total_dist < min_total_dist) {
                    min_total_dist = total_dist;
                    node->center_idx = candidates[i];
                }
            }
        }
    }

    // Rest of the function remains the same...
    size_t furthest_idx = find_furthest_point(indices, node->center_idx);
    
    std::vector<size_t> left_indices, right_indices;
    left_indices.push_back(node->center_idx);
    
    const float* center = m_data + node->center_idx * m_vector_size;
    const float* furthest = m_data + furthest_idx * m_vector_size;
    
    for (size_t idx : indices) {
        if (idx == node->center_idx) continue;
        
        const float* point = m_data + idx * m_vector_size;
        float dist_to_center = cosine_distance(center, point, m_vector_size);
        float dist_to_furthest = cosine_distance(furthest, point, m_vector_size);
        
        if (dist_to_center <= dist_to_furthest) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }
    
    if (std::find(left_indices.begin(), left_indices.end(), furthest_idx) == left_indices.end() &&
        std::find(right_indices.begin(), right_indices.end(), furthest_idx) == right_indices.end()) {
        right_indices.push_back(furthest_idx);
    }
    
    node->points = indices;
    node->radius = compute_radius(indices, node->center_idx);
    
    if (!left_indices.empty()) {
        build_tree(node->left, left_indices);
    }
    if (!right_indices.empty()) {
        build_tree(node->right, right_indices);
    }
}

void AdvancedKNNSearch::search_ball_tree(const BallNode* node,
                                        const float* query,
                                        std::priority_queue<std::pair<float, size_t>>& results,
                                        float& worst_dist,
                                        int k) const {
    if (!node) return;

    // Compute distance to center
    float dist_to_center = cosine_distance(query, m_data + node->center_idx * m_vector_size, m_vector_size);
    
    // If this is a leaf node, check all points
    if (!node->left && !node->right) {
        for (size_t idx : node->points) {
            float dist = cosine_distance(query, m_data + idx * m_vector_size, m_vector_size);
            
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
        return;
    }
    
    // Check if we can prune this subtree
    if (results.size() == static_cast<size_t>(k) && 
        dist_to_center - node->radius > worst_dist) {
        return;
    }
    
    // Recursively search both subtrees
    // Search the closer subtree first
    if (node->left && node->right) {
        float dist_to_left = cosine_distance(query, 
            m_data + node->left->center_idx * m_vector_size, m_vector_size);
        float dist_to_right = cosine_distance(query, 
            m_data + node->right->center_idx * m_vector_size, m_vector_size);
        
        if (dist_to_left < dist_to_right) {
            search_ball_tree(node->left.get(), query, results, worst_dist, k);
            search_ball_tree(node->right.get(), query, results, worst_dist, k);
        } else {
            search_ball_tree(node->right.get(), query, results, worst_dist, k);
            search_ball_tree(node->left.get(), query, results, worst_dist, k);
        }
    } else {
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