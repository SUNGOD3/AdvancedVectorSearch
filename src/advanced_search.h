// src/advanced_search.h
#pragma once
#include <vector>
#include <memory>
#include <queue>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

class BaseAdvancedSearch {
public:
    virtual ~BaseAdvancedSearch() = default;
    virtual py::array_t<int> search(py::array_t<float> query, int k) = 0;
protected:
    static float cosine_distance(const float* a, const float* b, size_t size);
    float* m_data;
    size_t m_num_vectors;
    size_t m_vector_size;
};

class AdvancedLinearSearch : public BaseAdvancedSearch {
public:
    AdvancedLinearSearch(py::array_t<float> vectors);
    ~AdvancedLinearSearch();
    
    AdvancedLinearSearch(const AdvancedLinearSearch&) = delete;
    AdvancedLinearSearch& operator=(const AdvancedLinearSearch&) = delete;
    
    py::array_t<int> search(py::array_t<float> query, int k) override;
private:
    void parallel_sort(std::pair<float, size_t>* distances, int k);
};

class AdvancedKNNSearch : public BaseAdvancedSearch {
public:
    AdvancedKNNSearch(py::array_t<float> vectors);
    ~AdvancedKNNSearch();
    
    AdvancedKNNSearch(const AdvancedKNNSearch&) = delete;
    AdvancedKNNSearch& operator=(const AdvancedKNNSearch&) = delete;
    
    py::array_t<int> search(py::array_t<float> query, int k) override;

protected:
    struct Node {
        size_t idx;
        const float* pivot;
        size_t split_dim;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        
        Node() : idx(0), pivot(nullptr), split_dim(0) {}
    };

private:    
    std::unique_ptr<Node> root;
    void build_tree(std::unique_ptr<Node>& node, std::vector<size_t>& indices, int depth);
    void search_tree(const Node* node, const float* query,
                    std::priority_queue<std::pair<float, size_t>>& results,
                    float& worst_dist, int k) const;
};