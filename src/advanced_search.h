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
    struct BallNode {
        size_t center_idx;       // Index of the center vector
        float radius;           // Radius of the ball
        std::vector<size_t> points;  // Points contained in this ball
        std::unique_ptr<BallNode> left;
        std::unique_ptr<BallNode> right;
        
        BallNode() : center_idx(0), radius(0.0f) {}
    };

private:    
    std::unique_ptr<BallNode> root;
    
    // Helper functions for ball tree construction and search
    void build_tree(std::unique_ptr<BallNode>& node, std::vector<size_t>& indices);
    float compute_radius(const std::vector<size_t>& indices, size_t center_idx);
    size_t find_furthest_point(const std::vector<size_t>& indices, size_t center_idx);
    void search_ball_tree(const BallNode* node, 
                         const float* query,
                         std::priority_queue<std::pair<float, size_t>>& results,
                         float& worst_dist,
                         int k) const;
};