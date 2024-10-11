#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "advanced_search.h"

namespace py = pybind11;

PYBIND11_MODULE(advanced_search_cpp, m) {
    py::class_<AdvancedSearch>(m, "AdvancedSearch")
        .def(py::init<py::array_t<double>>())
        .def("search", &AdvancedSearch::search);
}