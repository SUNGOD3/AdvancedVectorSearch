#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class AdvancedSearch {
public:
    AdvancedSearch(py::array_t<float> vectors);
    py::array_t<int> search(py::array_t<float> query, int k);

private:
    std::vector<std::vector<float>> m_vectors;
    static float cosine_distance(const std::vector<float>& a, const std::vector<float>& b);
};