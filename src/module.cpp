// src/module.cpp
#include "advanced_search.h"

namespace py = pybind11;

PYBIND11_MODULE(advanced_search_cpp, m) {
    py::class_<BaseAdvancedSearch>(m, "BaseAdvancedSearch");
    
    py::class_<AdvancedLinearSearch, BaseAdvancedSearch>(m, "AdvancedLinearSearch")
        .def(py::init<py::array_t<float>, std::string>())
        .def("search", &AdvancedLinearSearch::search);
        
    py::class_<AdvancedKNNSearch, BaseAdvancedSearch>(m, "AdvancedKNNSearch")
        .def(py::init<py::array_t<float>, std::string>())
        .def("search", &AdvancedKNNSearch::search);
}