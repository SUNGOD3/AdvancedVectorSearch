// src/advanced_search.h
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

class AdvancedLinearSearch : public BaseAdvancedSearch {
public:
    AdvancedLinearSearch(py::array_t<float> vectors, const std::string& metric = "cosine");
    ~AdvancedLinearSearch();
    
    AdvancedLinearSearch(const AdvancedLinearSearch&) = delete;
    AdvancedLinearSearch& operator=(const AdvancedLinearSearch&) = delete;
    
    py::array_t<int> search(py::array_t<float> query, int k) override;
};

class AdvancedKNNSearch : public BaseAdvancedSearch {
protected:
    struct BallNode {
        size_t center_idx;    // Index of the center vector
        float radius;         // Radius of the ball
        size_t* points;       // Points contained in this ball
        size_t num_points;    // Number of points in this node
        std::unique_ptr<BallNode> left;
        std::unique_ptr<BallNode> right;
        
        BallNode() : center_idx(0), radius(0.0f), points(nullptr), num_points(0) {}
        ~BallNode() { delete[] points; }
        
        // Prevent copying
        BallNode(const BallNode&) = delete;
        BallNode& operator=(const BallNode&) = delete;
        
        // Allow moving
        BallNode(BallNode&& other) noexcept
            : center_idx(other.center_idx)
            , radius(other.radius)
            , points(other.points)
            , num_points(other.num_points)
            , left(std::move(other.left))
            , right(std::move(other.right)) {
            other.points = nullptr;
            other.num_points = 0;
        }
        
        BallNode& operator=(BallNode&& other) noexcept {
            if (this != &other) {
                delete[] points;
                center_idx = other.center_idx;
                radius = other.radius;
                points = other.points;
                num_points = other.num_points;
                left = std::move(other.left);
                right = std::move(other.right);
                other.points = nullptr;
                other.num_points = 0;
            }
            return *this;
        }
    };

private:
    std::unique_ptr<BallNode> root;
    
    // Helper functions for ball tree construction and search
    void build_tree(std::unique_ptr<BallNode>& node, size_t* indices, size_t num_indices);
    float compute_radius(const size_t* indices, size_t num_indices, size_t center_idx);
    size_t find_furthest_point(const size_t* indices, size_t num_indices, size_t center_idx);
    void search_ball_tree(const BallNode* node, 
                         const float* query,
                         std::pair<float, size_t>* results,
                         size_t& result_size,
                         float& worst_dist,
                         size_t k) const;

public:
    AdvancedKNNSearch(py::array_t<float> vectors, const std::string& metric = "cosine");
    ~AdvancedKNNSearch();
    
    AdvancedKNNSearch(const AdvancedKNNSearch&) = delete;
    AdvancedKNNSearch& operator=(const AdvancedKNNSearch&) = delete;
    
    py::array_t<int> search(py::array_t<float> query, int k) override;
};


// src/advanced_search_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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