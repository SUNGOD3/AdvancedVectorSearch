#include "base_search.h"

float BaseAdvancedSearch::inner_product_distance(const float* a, const float* b, size_t size) {
    float dot = 0.0;
    
    #pragma omp simd reduction(+:dot)
    for (size_t i = 0; i < size; ++i) {
        dot += a[i] * b[i];
    }
    
    return -dot; // Negate for sorting (want highest dot product first)
}

float BaseAdvancedSearch::cosine_distance(const float* a, const float* b, size_t size, float norm_a, float norm_b) {
    float dot = 0.0;
    
    #pragma omp simd reduction(+:dot)
    for (size_t i = 0; i < size; ++i) {
        dot += a[i] * b[i];
    }

    return norm_a * norm_b / (dot * dot); // Precomputed norms optimization
}


float BaseAdvancedSearch::l2_distance(const float* a, const float* b, size_t size) {
    float sum = 0.0f;
    
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum; // sqrt(sum) is not necessary for ranking
}

float BaseAdvancedSearch::l2_distance_early_exit(const float* a, const float* b, size_t size, float threshold) {
    float sum = 0.0f;
    const size_t BATCH_SIZE = (size >> 3);
    
    for (size_t i = 0; i < size; i += BATCH_SIZE) {
        float batch_sum = 0.0f;
        size_t end = std::min(i + BATCH_SIZE, size);
        
        #pragma omp simd reduction(+:batch_sum)
        for (size_t j = i; j < end; ++j) {
            float diff = a[j] - b[j];
            batch_sum += diff * diff;
        }
        
        sum += batch_sum;
        if (sum > threshold) {
            return sum;
        }
    }
    
    return sum;
}

void BaseAdvancedSearch::parallel_sort(std::pair<float, size_t>* distances, int k) {
    int num_threads = 4;
    int block_size = (k + num_threads - 1) / num_threads;
    
    #pragma omp parallel for 
    for (int i = 0; i < k; i += block_size) {
        int block_end = std::min(i + block_size, k);
        std::sort(distances + i, distances + block_end);
    }
    
    for (int merge_size = block_size; merge_size < k; merge_size *= 2) {
        #pragma omp parallel for schedule(static)
        for (int i = 2 * merge_size; i < k; i += 2 * merge_size) {
            std::inplace_merge(distances + i - 2 * merge_size, distances + (i - merge_size), distances + i);
        }
    }
}

void BaseAdvancedSearch::normalize(float* data, size_t num_vectors, size_t vector_size) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_vectors; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < vector_size; ++j) {
            norm += data[i * vector_size + j] * data[i * vector_size + j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < vector_size; ++j) {
            data[i * vector_size + j] /= norm;
        }
    }
}