#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class AdvancedSearch {
public:
    AdvancedSearch(py::array_t<double> vectors);
    py::array_t<int> search(py::array_t<double> query, int k);

private:
    std::vector<std::vector<double>> m_vectors;
    static double cosine_distance(const std::vector<double>& a, const std::vector<double>& b);
};