#include "knn_search.h"

// Advanced KNN Search Implementation
AdvancedKNNSearch::AdvancedKNNSearch(py::array_t<float> vectors, const std::string& metric) {
    if (metric == "l2") {
        m_metric = DistanceMetric::L2;
    } else if (metric == "inner_product") {
        m_metric = DistanceMetric::INNER_PRODUCT;
    } else if (metric == "cosine") {
        m_metric = DistanceMetric::COSINE;
    } else {
        throw std::runtime_error("Invalid distance metric");
    }
    py::buffer_info buf = vectors.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    m_num_vectors = buf.shape[0];
    m_vector_size = buf.shape[1];
    
    size_t total_size = m_num_vectors * m_vector_size;
    m_data = new float[total_size];
    std::memcpy(m_data, buf.ptr, sizeof(float) * total_size);

    // Normalize vectors for inner product and cosine distance
    if (m_metric == DistanceMetric::INNER_PRODUCT || m_metric == DistanceMetric::COSINE) {
        normalize(m_data, m_num_vectors, m_vector_size);
    }

    // Initialize indices array
    size_t* indices = new size_t[m_num_vectors];
    std::iota(indices, indices + m_num_vectors, 0); // Fill with 0, 1, 2 ...
    
    build_tree(root, indices, m_num_vectors);
    delete[] indices;
}

float AdvancedKNNSearch::compute_radius(const size_t* indices, size_t num_indices, size_t center_idx) {
    float max_dist = 0.0f;
    const float* center = m_data + center_idx * m_vector_size;
    
    for (size_t i = 0; i < num_indices; ++i) {
        if (indices[i] == center_idx) continue;
        const float* point = m_data + indices[i] * m_vector_size;
        float dist = compute_distance(center, point, m_vector_size);
        max_dist = std::max(max_dist, dist);
    }
    return max_dist;
}

void AdvancedKNNSearch::build_tree(std::unique_ptr<BallNode>& node, size_t* indices, size_t num_indices) {
    if (num_indices == 0) return;
    node = std::make_unique<BallNode>();
    
    if (num_indices <= 128) {
        node->points = new size_t[num_indices];
        node->num_points = num_indices;
        std::memcpy(node->points, indices, sizeof(size_t) * num_indices);
        node->center_idx = indices[0];
        node->radius = compute_radius(indices, num_indices, node->center_idx);
        return;
    }

    const size_t sample_size = std::min(size_t(512), num_indices);
    float* mean = new float[m_vector_size]();
    size_t* sampled_indices = new size_t[sample_size];
    
    for (size_t i = 0; i < sample_size; ++i) {
        size_t rand_idx = i < num_indices ? i : rand() % num_indices;
        sampled_indices[i] = indices[rand_idx];
        
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
    
    for (size_t i = 0; i < sample_size; ++i) {
        const float* point = m_data + sampled_indices[i] * m_vector_size;
        float dist = 0.0f;
        
        for (size_t j = 0; j < m_vector_size; ++j) {
            float diff = point[j] - mean[j];
            dist += diff * diff;
        }
        
        if (dist < min_distance) {
            min_distance = dist;
            center_idx = sampled_indices[i];
        }
    }
    
    delete[] mean;
    delete[] sampled_indices;
    
    node->center_idx = center_idx;
    
    std::pair<float, size_t>* distances = new std::pair<float, size_t>[num_indices];
    size_t distance_count = 0;
    
    const float* center_point = m_data + center_idx * m_vector_size;
    for (size_t i = 0; i < num_indices; ++i) {
        if (indices[i] == center_idx) continue;
        
        const float* point = m_data + indices[i] * m_vector_size;
        float dist = compute_distance(center_point, point, m_vector_size);
        distances[distance_count++] = {dist, indices[i]};
    }
    
    size_t mid = distance_count / 2;
    std::nth_element(distances, distances + mid, distances + distance_count);
    
    size_t* left_indices = new size_t[distance_count + 1];
    size_t* right_indices = new size_t[distance_count];
    size_t left_count = 1, right_count = 0;
    
    left_indices[0] = center_idx;
    
    for (size_t i = 0; i < distance_count; ++i) {
        if (i < mid) {
            left_indices[left_count++] = distances[i].second;
        } else {
            right_indices[right_count++] = distances[i].second;
        }
    }
    
    delete[] distances;
    
    node->radius = compute_radius(indices, num_indices, center_idx);
    node->points = new size_t[num_indices];
    node->num_points = num_indices;
    std::memcpy(node->points, indices, sizeof(size_t) * num_indices);
    
    if (left_count > 0) build_tree(node->left, left_indices, left_count);
    if (right_count > 0) build_tree(node->right, right_indices, right_count);
    
    delete[] left_indices;
    delete[] right_indices;
}

void AdvancedKNNSearch::search_ball_tree(const BallNode* node,
                                        const float* query,
                                        std::pair<float, size_t>* results,
                                        size_t& result_size,
                                        float& worst_dist,
                                        size_t k) const {
    if (!node) return;

    // Process leaf node
    if (!node->left && !node->right) {
    
        #pragma omp parallel for
        for (size_t i = 0; i < node->num_points; ++i) {
            size_t idx = node->points[i];
            const float* point = m_data + idx * m_vector_size;
            
            float dist = (m_metric == DistanceMetric::L2) 
                ? l2_distance_early_exit(query, point, m_vector_size, worst_dist) 
                : compute_distance(query, point, m_vector_size);
                
            #pragma omp critical
            {
                if (result_size < k) {
                    results[result_size++] = {dist, idx};
                    if (result_size == k) {
                        std::make_heap(results, results + k);
                        worst_dist = results[0].first;
                    }
                } else if (dist < worst_dist) {
                    std::pop_heap(results, results + k);
                    results[k-1] = {dist, idx};
                    std::push_heap(results, results + k);
                    worst_dist = results[0].first;
                }
            }
        }
        return;
    }
    
    if (node->left) search_ball_tree(node->left.get(), query, results, result_size, worst_dist, k);
    if (node->right) search_ball_tree(node->right.get(), query, results, result_size, worst_dist, k);
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
    
    // Use parallel_sort if k >= m_num_vectors / 2
    if (k >= static_cast<int>(m_num_vectors/2)) {
        std::pair<float, size_t>* distances = new std::pair<float, size_t>[m_num_vectors];
        
        #pragma omp parallel for
        for (size_t i = 0; i < m_num_vectors; ++i) {
            distances[i] = {compute_distance(query_ptr, m_data + i * m_vector_size, m_vector_size), i};
        }
        
        k = std::min(k, static_cast<int>(m_num_vectors));
        std::nth_element(distances, distances + k, distances + m_num_vectors);
        parallel_sort(distances, k);
        
        py::array_t<int> result(k);
        auto result_ptr = result.mutable_data();
        
        #pragma omp parallel for
        for (int i = 0; i < k; ++i) {
            result_ptr[i] = distances[i].second;
        }
        
        delete[] distances;
        return result;
    }
    
    std::pair<float, size_t>* results = new std::pair<float, size_t>[k];
    size_t result_size = 0;
    float worst_dist = std::numeric_limits<float>::max();
    
    search_ball_tree(root.get(), query_ptr, results, result_size, worst_dist, k);
    
    std::sort_heap(results, results + result_size);
    
    py::array_t<int> result(result_size);
    auto result_ptr = result.mutable_data();
    for (size_t i = 0; i < result_size; ++i) {
        result_ptr[i] = results[i].second;
    }
    
    delete[] results;
    return result;
}

AdvancedKNNSearch::~AdvancedKNNSearch() {
    // Base class destructor will handle m_data
    // root's destructor will automatically clean up the entire tree
    // through the recursive destruction of unique_ptrs and BallNode destructor
}
