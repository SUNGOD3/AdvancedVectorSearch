#pragma once
#include "base_search.h"

class AdvancedLinearSearch : public BaseAdvancedSearch {
public:
    AdvancedLinearSearch(py::array_t<float> vectors, const std::string& metric = "cosine");
    ~AdvancedLinearSearch();
    
    AdvancedLinearSearch(const AdvancedLinearSearch&) = delete;
    AdvancedLinearSearch& operator=(const AdvancedLinearSearch&) = delete;
    
    py::array_t<int> search(py::array_t<float> query, int k) override;
};