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
    
    // Optimization: Use vectorized block processing with explicit SIMD optimization
    const size_t block_size = 256;
    const size_t num_blocks = (indices.size() + block_size - 1) / block_size;
    
    #pragma omp parallel
    {
        float thread_max_dist = 0.0f;
        
        #pragma omp for nowait schedule(static)
        for (size_t block = 0; block < num_blocks; ++block) {
            const size_t start_idx = block * block_size;
            const size_t end_idx = std::min(start_idx + block_size, indices.size());
            
            float block_max_dist = 0.0f;
            
            for (size_t i = start_idx; i < end_idx; ++i) {
                if (indices[i] == center_idx) continue;
                
                const float* point = m_data + indices[i] * m_vector_size;
                float dot = 0.0f, denom_a = 0.0f, denom_b = 0.0f;
                
                // Explicit SIMD optimization for distance calculation
                #pragma omp simd reduction(+:dot,denom_a,denom_b) aligned(center,point:32)
                for (size_t j = 0; j < m_vector_size; ++j) {
                    dot += center[j] * point[j];
                    denom_a += center[j] * center[j];
                    denom_b += point[j] * point[j];
                }
                
                float dist = 1.0f - dot / std::sqrt(denom_a * denom_b);
                block_max_dist = std::max(block_max_dist, dist);
            }
            
            thread_max_dist = std::max(thread_max_dist, block_max_dist);
        }
        
        #pragma omp critical
        {
            max_dist = std::max(max_dist, thread_max_dist);
        }
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

    // Optimize center selection using Principal Direction strategy
    const size_t sample_size = std::min(size_t(100), indices.size());
    std::vector<float> mean(m_vector_size, 0.0f);
    std::vector<float> principal_direction(m_vector_size, 0.0f);
    
    // Calculate mean of sampled points
    #pragma omp parallel
    {
        std::vector<float> local_mean(m_vector_size, 0.0f);
        
        #pragma omp for nowait schedule(static)
        for (size_t i = 0; i < sample_size; ++i) {
            const float* point = m_data + indices[i] * m_vector_size;
            #pragma omp simd
            for (size_t j = 0; j < m_vector_size; ++j) {
                local_mean[j] += point[j];
            }
        }
        
        #pragma omp critical
        {
            #pragma omp simd
            for (size_t j = 0; j < m_vector_size; ++j) {
                mean[j] += local_mean[j];
            }
        }
    }
    
    const float scale = 1.0f / sample_size;
    #pragma omp simd
    for (size_t j = 0; j < m_vector_size; ++j) {
        mean[j] *= scale;
    }
    
    // Find the principal direction using power iteration
    std::vector<float> temp_vector(m_vector_size);
    for (size_t iter = 0; iter < 3; ++iter) { // Usually 3 iterations is enough
        std::fill(temp_vector.begin(), temp_vector.end(), 0.0f);
        
        #pragma omp parallel
        {
            std::vector<float> local_direction(m_vector_size, 0.0f);
            
            #pragma omp for schedule(static)
            for (size_t i = 0; i < sample_size; ++i) {
                const float* point = m_data + indices[i] * m_vector_size;
                float proj = 0.0f;
                
                // Calculate projection
                #pragma omp simd reduction(+:proj)
                for (size_t j = 0; j < m_vector_size; ++j) {
                    proj += (point[j] - mean[j]) * principal_direction[j];
                }
                
                // Update direction
                #pragma omp simd
                for (size_t j = 0; j < m_vector_size; ++j) {
                    local_direction[j] += proj * (point[j] - mean[j]);
                }
            }
            
            #pragma omp critical
            {
                #pragma omp simd
                for (size_t j = 0; j < m_vector_size; ++j) {
                    temp_vector[j] += local_direction[j];
                }
            }
        }
        
        // Normalize
        float norm = 0.0f;
        #pragma omp simd reduction(+:norm)
        for (size_t j = 0; j < m_vector_size; ++j) {
            norm += temp_vector[j] * temp_vector[j];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-10) {
            #pragma omp simd
            for (size_t j = 0; j < m_vector_size; ++j) {
                principal_direction[j] = temp_vector[j] / norm;
            }
        }
    }
    
    // Select center point based on projected values
    float best_center_score = std::numeric_limits<float>::max();
    node->center_idx = indices[0];
    
    #pragma omp parallel
    {
        float local_best_score = std::numeric_limits<float>::max();
        size_t local_best_idx = indices[0];
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < sample_size; ++i) {
            const float* point = m_data + indices[i] * m_vector_size;
            float score = 0.0f;
            
            for (size_t j = 0; j < sample_size; ++j) {
                const float* other = m_data + indices[j] * m_vector_size;
                float dist = cosine_distance(point, other, m_vector_size);
                score += dist;
            }
            
            if (score < local_best_score) {
                local_best_score = score;
                local_best_idx = indices[i];
            }
        }
        
        #pragma omp critical
        {
            if (local_best_score < best_center_score) {
                best_center_score = local_best_score;
                node->center_idx = local_best_idx;
            }
        }
    }
    
    // Partition points based on distance to center
    std::vector<size_t> left_indices, right_indices;
    const float* center = m_data + node->center_idx * m_vector_size;
    
    // Reserve space to avoid reallocations
    left_indices.reserve(indices.size() / 2);
    right_indices.reserve(indices.size() / 2);
    
    // Compute projections and split based on median
    std::vector<std::pair<float, size_t>> projections(indices.size());
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < indices.size(); ++i) {
        float proj = 0.0f;
        const float* point = m_data + indices[i] * m_vector_size;
        
        #pragma omp simd reduction(+:proj)
        for (size_t j = 0; j < m_vector_size; ++j) {
            proj += (point[j] - mean[j]) * principal_direction[j];
        }
        projections[i] = {proj, indices[i]};
    }
    
    // Find median
    size_t mid = projections.size() / 2;
    std::nth_element(projections.begin(), projections.begin() + mid, projections.end());
    
    // Split points
    left_indices.push_back(node->center_idx);
    for (size_t i = 0; i < projections.size(); ++i) {
        if (projections[i].second == node->center_idx) continue;
        
        if (i < mid) {
            left_indices.push_back(projections[i].second);
        } else {
            right_indices.push_back(projections[i].second);
        }
    }
    
    // Update node info
    node->radius = compute_radius(indices, node->center_idx);
    node->points = indices;
    
    // Build subtrees
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

    // Calculate distance to center with SIMD optimization
    const float* center = m_data + node->center_idx * m_vector_size;
    float dot = 0.0f, denom_a = 0.0f, denom_b = 0.0f;
    
    #pragma omp simd reduction(+:dot,denom_a,denom_b)
    for (size_t i = 0; i < m_vector_size; ++i) {
        dot += center[i] * query[i];
        denom_a += center[i] * center[i];
        denom_b += query[i] * query[i];
    }
    float dist_to_center = 1.0f - dot / std::sqrt(denom_a * denom_b);
    
    // Early pruning check
    if (results.size() == static_cast<size_t>(k) && 
        dist_to_center - node->radius > worst_dist) {
        return;
    }
    
    // Leaf node processing
    if (!node->left && !node->right) {
        // Process points in small batches to maintain cache efficiency
        const size_t batch_size = 16;
        const size_t num_points = node->points.size();
        
        for (size_t batch_start = 0; batch_start < num_points; batch_start += batch_size) {
            const size_t batch_end = std::min(batch_start + batch_size, num_points);
            
            // Process each point in the batch
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = node->points[i];
                const float* point = m_data + idx * m_vector_size;
                
                float dot = 0.0f, denom_a = 0.0f;
                
                #pragma omp simd reduction(+:dot,denom_a)
                for (size_t j = 0; j < m_vector_size; ++j) {
                    dot += point[j] * query[j];
                    denom_a += point[j] * point[j];
                }
                
                float dist = 1.0f - dot / std::sqrt(denom_a * denom_b);
                
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
    
    // Recursively search both subtrees
    if (node->left && node->right) {
        // Calculate distances to both child centers
        const float* left_center = m_data + node->left->center_idx * m_vector_size;
        const float* right_center = m_data + node->right->center_idx * m_vector_size;
        
        float left_dot = 0.0f, left_norm = 0.0f;
        float right_dot = 0.0f, right_norm = 0.0f;
        
        #pragma omp simd reduction(+:left_dot,left_norm,right_dot,right_norm)
        for (size_t i = 0; i < m_vector_size; ++i) {
            left_dot += left_center[i] * query[i];
            left_norm += left_center[i] * left_center[i];
            right_dot += right_center[i] * query[i];
            right_norm += right_center[i] * right_center[i];
        }
        
        float dist_to_left = 1.0f - left_dot / std::sqrt(left_norm * denom_b);
        float dist_to_right = 1.0f - right_dot / std::sqrt(right_norm * denom_b);
        
        // Visit closer node first
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