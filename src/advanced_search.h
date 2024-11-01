#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class AdvancedSearch {
public:
    AdvancedSearch(py::array_t<float> vectors);
    ~AdvancedSearch();  
    
    // Disable copy and move operations
    AdvancedSearch(const AdvancedSearch&) = delete;
    AdvancedSearch& operator=(const AdvancedSearch&) = delete;
    
    py::array_t<int> search(py::array_t<float> query, int k);

private:
    float* m_data;           
    size_t m_num_vectors;    
    size_t m_vector_size;    
    
    static float cosine_distance(const float* a, const float* b, size_t size);
};