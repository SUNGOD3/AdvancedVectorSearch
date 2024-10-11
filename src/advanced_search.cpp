#include "advanced_search.h"
#include <algorithm>
#include <cmath>

AdvancedSearch::AdvancedSearch(py::array_t<float> vectors) {
    py::buffer_info buf = vectors.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    size_t num_vectors = buf.shape[0];
    size_t vector_size = buf.shape[1];
    float *ptr = static_cast<float *>(buf.ptr);

    m_vectors.reserve(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        m_vectors.emplace_back(ptr + i * vector_size, ptr + (i + 1) * vector_size);
    }
}

py::array_t<int> AdvancedSearch::search(py::array_t<float> query, int k) {
    py::buffer_info buf = query.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    
    std::vector<float> query_vec(buf.shape[0]);
    std::memcpy(query_vec.data(), buf.ptr, sizeof(float) * buf.shape[0]);

    std::vector<std::pair<float, size_t>> distances;
    distances.reserve(m_vectors.size());

    for (size_t i = 0; i < m_vectors.size(); ++i) {
        float dist = cosine_distance(query_vec, m_vectors[i]);
        distances.emplace_back(dist, i);
    }

    // God bless C++
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

    py::array_t<int> result(k);
    auto result_buf = result.mutable_unchecked<1>();
    for (int i = 0; i < k; ++i) {
        result_buf(i) = static_cast<int>(distances[i].second);
    }

    return result;
}

float AdvancedSearch::cosine_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        denom_a += a[i] * a[i];
        denom_b += b[i] * b[i];
    }
    return 1.0 - (dot / (std::sqrt(denom_a) * std::sqrt(denom_b)));
}