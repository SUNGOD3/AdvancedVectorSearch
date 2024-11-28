# benchmarks/run_benchmarks.py

import time
import json
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search import LinearSearch, FaissSearch, AdvancedLinearSearch, AdvancedKNNSearch
from src.data_generator import generate_random_vectors

def evaluate_accuracy(ground_truth, predicted):
    """
    Evaluate the accuracy of the search results.
    
    :param ground_truth: List of indices of the true nearest vectors
    :param predicted: List of indices returned by the search method
    :return: Accuracy score (between 0.0 and 1.0)
    """
    ground_truth_set = set(ground_truth[0])
    predicted_set = set(predicted[0])
    common_elements = ground_truth_set & predicted_set
    return len(common_elements) / len(ground_truth_set)

def run_benchmark(search_method, vectors, queries, k):
    start_time = time.time()
    results = []
    for query in queries:
        results.append(search_method.search(query, k))
    end_time = time.time()
    return results, end_time - start_time

def benchmark_dataset(vectors, queries, k, dataset_name, metric):
    """
    Run benchmark for a specific dataset and metric.
    
    :param vectors: Dataset vectors
    :param queries: Query vectors
    :param k: Number of nearest neighbors
    :param dataset_name: Name of the dataset
    :param metric: Distance metric to use
    :return: Dictionary containing benchmark results
    """
    print(f"\nRunning benchmark for {dataset_name} dataset with {metric} metric:")
    print(f"Vectors: {len(vectors)}, Queries: {len(queries)}, k={k}")

    # Initialize all search methods
    linear_search_training_start = time.time()
    linear_search = LinearSearch(vectors, metric=metric)
    linear_search_training_time = time.time() - linear_search_training_start

    advanced_linear_training_start = time.time()
    advanced_linear = AdvancedLinearSearch(vectors, metric=metric)
    advanced_linear_training_time = time.time() - advanced_linear_training_start

    advanced_knn_training_start = time.time()
    advanced_knn = AdvancedKNNSearch(vectors, metric=metric)
    advanced_knn_training_time = time.time() - advanced_knn_training_start

    faiss_search_training_start = time.time()
    faiss_search = FaissSearch(vectors, metric=metric)
    faiss_search_training_time = time.time() - faiss_search_training_start

    # Run benchmarks
    linear_results, linear_time = run_benchmark(linear_search, vectors, queries, k)
    advanced_linear_results, advanced_linear_time = run_benchmark(advanced_linear, vectors, queries, k)
    advanced_knn_results, advanced_knn_time = run_benchmark(advanced_knn, vectors, queries, k)
    faiss_results, faiss_time = run_benchmark(faiss_search, vectors, queries, k)

    # Calculate accuracy against linear search (ground truth)
    advanced_linear_accuracy = evaluate_accuracy(linear_results, advanced_linear_results)
    advanced_knn_accuracy = evaluate_accuracy(linear_results, advanced_knn_results)
    faiss_accuracy = evaluate_accuracy(linear_results, faiss_results)

    # Print results
    print(f"Linear search time: {linear_time:.4f} s, accuracy: 1.0 (ground truth), training time: {linear_search_training_time:.4f} s")
    print(f"Advanced linear search time: {advanced_linear_time:.4f} s, accuracy: {advanced_linear_accuracy:.4f}, training time: {advanced_linear_training_time:.4f} s")
    print(f"Advanced KNN search time: {advanced_knn_time:.4f} s, accuracy: {advanced_knn_accuracy:.4f}, training time: {advanced_knn_training_time:.4f} s")
    print(f"Faiss search time: {faiss_time:.4f} s, accuracy: {faiss_accuracy:.4f}, training time: {faiss_search_training_time:.4f} s")

    return {
        "linear_search": {
            "training_time": linear_search_training_time,
            "time": linear_time
        },
        "advanced_linear_search": {
            "training_time": advanced_linear_training_time,
            "time": advanced_linear_time,
            "accuracy": advanced_linear_accuracy
        },
        "advanced_knn_search": {
            "training_time": advanced_knn_training_time,
            "time": advanced_knn_time,
            "accuracy": advanced_knn_accuracy
        },
        "faiss_search": {
            "training_time": faiss_search_training_time,
            "time": faiss_time,
            "accuracy": faiss_accuracy
        }
    }

def benchmark_large_dataset(vectors, queries, k, dataset_name, metric):
    """
    Run benchmark for a large dataset and specific metric.
    
    :param vectors: Dataset vectors
    :param queries: Query vectors
    :param k: Number of nearest neighbors
    :param dataset_name: Name of the dataset
    :param metric: Distance metric to use
    :return: Dictionary containing benchmark results
    """
    print(f"\nRunning benchmark for {dataset_name} dataset with {metric} metric:")
    print(f"Vectors: {len(vectors)}, Queries: {len(queries)}, k={k}")

    # Initialize optimized search methods
    advanced_linear_training_start = time.time()
    advanced_linear = AdvancedLinearSearch(vectors, metric=metric)
    advanced_linear_training_time = time.time() - advanced_linear_training_start

    advanced_knn_training_start = time.time()
    advanced_knn = AdvancedKNNSearch(vectors, metric=metric)
    advanced_knn_training_time = time.time() - advanced_knn_training_start

    faiss_search_training_start = time.time()
    faiss_search = FaissSearch(vectors, metric=metric)
    faiss_search_training_time = time.time() - faiss_search_training_start

    # Run benchmarks
    advanced_linear_results, advanced_linear_time = run_benchmark(advanced_linear, vectors, queries, k)
    advanced_knn_results, advanced_knn_time = run_benchmark(advanced_knn, vectors, queries, k)
    faiss_results, faiss_time = run_benchmark(faiss_search, vectors, queries, k)

    # Calculate accuracy (comparing each method with others)
    advanced_linear_accuracy = evaluate_accuracy(advanced_linear_results, advanced_knn_results)
    advanced_knn_accuracy = evaluate_accuracy(advanced_knn_results, faiss_results)
    faiss_accuracy = evaluate_accuracy(faiss_results, advanced_linear_results)

    print(f"Advanced linear search time: {advanced_linear_time:.4f} s, similarity to Advanced KNN: {advanced_linear_accuracy:.4f}, training time: {advanced_linear_training_time:.4f} s")
    print(f"Advanced KNN search time: {advanced_knn_time:.4f} s, similarity to Faiss: {advanced_knn_accuracy:.4f}, training time: {advanced_knn_training_time:.4f} s")
    print(f"Faiss search time: {faiss_time:.4f} s, similarity to Advanced Linear: {faiss_accuracy:.4f}, training time: {faiss_search_training_time:.4f} s")

    return {
        "advanced_linear_search": {
            "training_time": advanced_linear_training_time,
            "time": advanced_linear_time,
            "accuracy": advanced_linear_accuracy
        },
        "advanced_knn_search": {
            "training_time": advanced_knn_training_time,
            "time": advanced_knn_time,
            "accuracy": advanced_knn_accuracy
        },
        "faiss_search": {
            "training_time": faiss_search_training_time,
            "time": faiss_time,
            "accuracy": faiss_accuracy
        }
    }

def main():
    np.random.seed(42)
    dimensions = 1024
    metrics = ["cosine", "l2", "inner_product"]

    datasets = {
        "small": {"num_vectors": 100, "num_queries": 10, "k": 10},
        "medium": {"num_vectors": 1000, "num_queries": 100, "k": 20},
        "large": {"num_vectors": 10000, "num_queries": 100, "k": 30},
    }

    large_datasets = {
        "extra_large": {"num_vectors": 50000, "num_queries": 200, "k": 50},
        "huge": {"num_vectors": 100000, "num_queries": 200, "k": 50},
        "huge_query": {"num_vectors": 100000, "num_queries": 200, "k": 10000},
        "huge_full_query": {"num_vectors": 100000, "num_queries": 200, "k": 100000},
    }

    results = {}

    # Benchmark regular datasets
    for dataset_name, params in datasets.items():
        vectors = generate_random_vectors(params["num_vectors"], dimensions)
        queries = generate_random_vectors(params["num_queries"], dimensions)
        
        results[dataset_name] = {}
        for metric in metrics:
            results[dataset_name][metric] = benchmark_dataset(
                vectors, queries, params["k"], dataset_name, metric
            )

    # Benchmark large datasets
    for dataset_name, params in large_datasets.items():
        vectors = generate_random_vectors(params["num_vectors"], dimensions)
        queries = generate_random_vectors(params["num_queries"], dimensions)
        
        results[dataset_name] = {}
        for metric in metrics:
            results[dataset_name][metric] = benchmark_large_dataset(
                vectors, queries, params["k"], dataset_name, metric
            )

    # Save results
    os.makedirs('benchmarks', exist_ok=True)
    with open('benchmarks/results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()