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
    std::vector<std::pair<float, size_t>> distances;
    distances.reserve(m_num_vectors);

    for (size_t i = 0; i < m_num_vectors; ++i) {
        float dist = cosine_distance(query_ptr, 
                                   m_data + i * m_vector_size, 
                                   m_vector_size);
        distances.emplace_back(dist, i);
    }

    k = std::min(k, static_cast<int>(m_num_vectors));
    std::partial_sort(distances.begin(), 
                     distances.begin() + k, 
                     distances.end());

    py::array_t<int> result(k);
    auto result_buf = result.mutable_unchecked<1>();
    for (int i = 0; i < k; ++i) {
        result_buf(i) = static_cast<int>(distances[i].second);
    }

    return result;
}

float AdvancedSearch::cosine_distance(const float* a, const float* b, size_t size) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    
    size_t i;
    for (i = 0; i + 4 <= size; i += 4) {
        dot += a[i] * b[i] + a[i+1] * b[i+1] + 
               a[i+2] * b[i+2] + a[i+3] * b[i+3];
        denom_a += a[i] * a[i] + a[i+1] * a[i+1] + 
                   a[i+2] * a[i+2] + a[i+3] * a[i+3];
        denom_b += b[i] * b[i] + b[i+1] * b[i+1] + 
                   b[i+2] * b[i+2] + b[i+3] * b[i+3];
    }
    
    for (; i < size; ++i) {
        dot += a[i] * b[i];
        denom_a += a[i] * a[i];
        denom_b += b[i] * b[i];
    }
    
    return 1.0 - (dot / (std::sqrt(denom_a) * std::sqrt(denom_b)));
}