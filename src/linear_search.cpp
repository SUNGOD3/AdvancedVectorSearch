#include "linear_search.h"

// Advanced Linear Search Implementation
AdvancedLinearSearch::AdvancedLinearSearch(py::array_t<float> vectors, const std::string& metric) {
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

    if (m_metric == DistanceMetric::COSINE) {
        m_norms = new float[m_num_vectors];

        #pragma omp parallel for
        for (size_t i = 0; i < m_num_vectors; ++i) {
            float norm = 0.0f;
            const float* vec = m_data + i * m_vector_size;
            
            #pragma omp simd reduction(+:norm)
            for (size_t j = 0; j < m_vector_size; ++j) {
                norm += vec[j] * vec[j];
            }
            
            m_norms[i] = norm;
        }
    }// else m_norms = nullptr;

}

AdvancedLinearSearch::~AdvancedLinearSearch() {
    delete[] m_data;
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

    // Compute query norm if using cosine distance
    
    if (m_metric == DistanceMetric::COSINE) {
        float query_norm = 0.0f;
        #pragma omp simd reduction(+:query_norm)
        for (size_t j = 0; j < m_vector_size; ++j) {
            query_norm += query_ptr[j] * query_ptr[j];
        }
        #pragma omp parallel for
        for (size_t i = 0; i < m_num_vectors; ++i) {
            distances[i] = {cosine_distance(query_ptr, m_data + i * m_vector_size, m_vector_size, query_norm, m_norms[i]), i};
        }
    }
    else if(m_metric == DistanceMetric::L2) {
        #pragma omp parallel for
        for (size_t i = 0; i < m_num_vectors; ++i) {
            distances[i] = {l2_distance(query_ptr, m_data + i * m_vector_size, m_vector_size), i};
        }
    }
    else{ // Inner Product
        #pragma omp parallel for
        for (size_t i = 0; i < m_num_vectors; ++i) {
            distances[i] = {inner_product_distance(query_ptr, m_data + i * m_vector_size, m_vector_size), i};
        }
    }

    k = std::min(k, static_cast<int>(m_num_vectors));
    std::nth_element(distances, distances + k, distances + m_num_vectors);
    parallel_sort(distances, k);

    py::array_t<int> result(k);
    auto result_ptr = static_cast<int*>(result.request().ptr); 

    #pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        result_ptr[i] = distances[i].second; 
    }


    delete[] distances;
    return result;
}