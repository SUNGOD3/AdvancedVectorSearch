#include "advanced_search.h"
#include <algorithm>
#include <cmath>
#include <cstring>

AdvancedSearch::AdvancedSearch(py::array_t<float> vectors) {
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

AdvancedSearch::~AdvancedSearch() {
    delete[] m_data;
}

py::array_t<int> AdvancedSearch::search(py::array_t<float> query, int k) {
    py::buffer_info buf = query.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    if (static_cast<size_t>(buf.shape[0]) != m_vector_size) {
        throw std::runtime_error("Query vector dimension mismatch");
    }
    
    const float* query_ptr = static_cast<float*>(buf.ptr);
    std::pair<float, size_t> distances[m_num_vectors];

    #pragma omp parallel for
    for (size_t i = 0; i < m_num_vectors; ++i) {
        distances[i] = {cosine_distance(query_ptr, m_data + i * m_vector_size, m_vector_size), i};
    }

    k = std::min(k, static_cast<int>(m_num_vectors));

    std::nth_element(distances, distances + k, distances + m_num_vectors);

    //std::sort(distances, distances + k);

    int num_threads = 4, size = k;
    int block_size = (size + num_threads - 1) / num_threads;
    
    // Step 1: Sort blocks
    #pragma omp parallel for 
    for (int i = 0; i < size; i += block_size) {
        int block_end = std::min(i + block_size, size);
        std::sort(distances + i, distances + block_end);
    }
    
    // Step 2: Merge blocks
    for (int merge_size = block_size; merge_size < size; merge_size *= 2) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < size; i += 2 * merge_size) {
            int mid = std::min(i + merge_size, size);
            int end = std::min(i + 2 * merge_size, size);
            if (end > mid) {
                std::inplace_merge(distances + i, distances + mid, distances + end);
            }
        }
    }
    
    py::array_t<int> result(k);
    #pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        result.mutable_at(i) = distances[i].second;
    }

    return result;
}

float AdvancedSearch::cosine_distance(const float* a, const float* b, size_t size) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    
    # pragma omp simd reduction(+:dot,denom_a,denom_b)
    for (size_t i = 0; i < size; ++i) {
        dot += a[i] * b[i];
        denom_a += a[i] * a[i];
        denom_b += b[i] * b[i];
    }

    return 1.0f - dot / std::sqrt(denom_a * denom_b);
}