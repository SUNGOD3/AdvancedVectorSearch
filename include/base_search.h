#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <omp.h>
#include <queue>
#include <vector>

namespace py = pybind11;

class BaseAdvancedSearch {
public:
    enum class DistanceMetric {
        COSINE,
        L2,
        INNER_PRODUCT
    };

    virtual ~BaseAdvancedSearch() = default;
    virtual py::array_t<int> search(py::array_t<float> query, int k) = 0;
    
protected:
    void normalize(float* data, size_t num_vectors, size_t vector_size);
    float compute_distance(const float* a, const float* b, size_t size) const {
        return (m_metric == DistanceMetric::L2) ? l2_distance(a, b, size) : inner_product_distance(a, b, size);
    }
    
    static float inner_product_distance(const float* a, const float* b, size_t size);
    static float cosine_distance(const float* a, const float* b, size_t size, float norm_a, float norm_b);
    static float l2_distance(const float* a, const float* b, size_t size);
    static float l2_distance_early_exit(const float* a, const float* b, size_t size, float threshold);
    static void parallel_sort(std::pair<float, size_t>* distances, int k);
    
    float* m_data;
    float* m_norms;
    size_t m_num_vectors;
    size_t m_vector_size;
    DistanceMetric m_metric;
};