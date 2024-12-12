#pragma once
#include "base_search.h"

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
    
    void build_tree(std::unique_ptr<BallNode>& node, size_t* indices, size_t num_indices);
    float compute_radius(const size_t* indices, size_t num_indices, size_t center_idx);
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