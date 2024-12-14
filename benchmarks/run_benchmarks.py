# benchmarks/run_benchmarks.py
import time
import json
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search import LinearSearch, FaissSearch, AdvancedLinearSearch, AdvancedKNNSearch, AdvancedHNSWSearch
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

def benchmark_dataset(vectors, queries, k, dataset_name, metric, dimension):
    """
    Run benchmark for a specific dataset and metric.
    
    :param vectors: Dataset vectors
    :param queries: Query vectors
    :param k: Number of nearest neighbors
    :param dataset_name: Name of the dataset
    :param metric: Distance metric to use
    :param dimension: Dimensionality of vectors
    :return: Dictionary containing benchmark results
    """
    print(f"\nRunning benchmark for {dataset_name} dataset with {metric} metric and dimension {dimension}:")
    print(f"Vectors: {len(vectors)}, Queries: {len(queries)}, k={k}")

    # Search method classes
    search_methods = [
        (LinearSearch, "linear_search"),
        (AdvancedLinearSearch, "advanced_linear_search"),
        (AdvancedKNNSearch, "advanced_knn_search"),
        (AdvancedHNSWSearch, "advanced_hnsw_search"),
        (FaissSearch, "faiss_search")
    ]

    results = {}
    for search_class, method_name in search_methods:
        training_start = time.time()
        search_instance = search_class(vectors, metric=metric)
        training_time = time.time() - training_start

        benchmark_results, search_time = run_benchmark(search_instance, vectors, queries, k)
        
        results[method_name] = {
            "training_time": training_time,
            "total_time": search_time,
            "avg_query_time": search_time / len(queries),  # New line
            "results": benchmark_results 
        }

        if method_name != "linear_search":
            results[method_name]["accuracy"] = evaluate_accuracy(
                results["linear_search"]["results"], 
                benchmark_results
            )

    print(f"Linear search time: {results['linear_search']['total_time']:.4f} s, avg query time: {results['linear_search']['avg_query_time'] * 1000:.6f} ms, accuracy: 1.0 (ground truth), training time: {results['linear_search']['training_time']:.4f} s")
    print(f"Advanced linear search time: {results['advanced_linear_search']['total_time']:.4f} s, avg query time: {results['advanced_linear_search']['avg_query_time'] * 1000:.6f} ms, accuracy: {results['advanced_linear_search']['accuracy']:.4f}, training time: {results['advanced_linear_search']['training_time']:.4f} s")
    print(f"Advanced KNN search time: {results['advanced_knn_search']['total_time']:.4f} s, avg query time: {results['advanced_knn_search']['avg_query_time'] * 1000:.6f} ms, accuracy: {results['advanced_knn_search']['accuracy']:.4f}, training time: {results['advanced_knn_search']['training_time']:.4f} s")
    print(f"Advanced HNSW search time: {results['advanced_hnsw_search']['total_time']:.4f} s, avg query time: {results['advanced_hnsw_search']['avg_query_time'] * 1000:.6f} ms, accuracy: {results['advanced_hnsw_search']['accuracy']:.4f}, training time: {results['advanced_hnsw_search']['training_time']:.4f} s")
    print(f"Faiss search time: {results['faiss_search']['total_time']:.4f} s, avg query time: {results['faiss_search']['avg_query_time'] * 1000:.6f} ms, accuracy: {results['faiss_search']['accuracy']:.4f}, training time: {results['faiss_search']['training_time']:.4f} s")

    return results

def benchmark_large_dataset(vectors, queries, k, dataset_name, metric, dimension):
    """
    Run benchmark for a large dataset and specific metric.
    
    :param vectors: Dataset vectors
    :param queries: Query vectors
    :param k: Number of nearest neighbors
    :param dataset_name: Name of the dataset
    :param metric: Distance metric to use
    :param dimension: Dimensionality of vectors
    :return: Dictionary containing benchmark results
    """
    print(f"\nRunning benchmark for {dataset_name} dataset with {metric} metric and dimension {dimension}:")
    print(f"Vectors: {len(vectors)}, Queries: {len(queries)}, k={k}")

    search_methods = [
        (AdvancedLinearSearch, "advanced_linear_search"),
        (AdvancedKNNSearch, "advanced_knn_search"),
        (FaissSearch, "faiss_search")
    ]

    results = {}
    for search_class, method_name in search_methods:
        training_start = time.time()
        search_instance = search_class(vectors, metric=metric)
        training_time = time.time() - training_start

        benchmark_results, search_time = run_benchmark(search_instance, vectors, queries, k)
        
        results[method_name] = {
            "training_time": training_time,
            "total_time": search_time,
            "avg_query_time": search_time / len(queries),  # New line
            "results": benchmark_results
        }

    results["advanced_linear_search"]["accuracy"] = evaluate_accuracy(
        results["advanced_linear_search"]["results"], 
        results["advanced_knn_search"]["results"]
    )
    results["advanced_knn_search"]["accuracy"] = evaluate_accuracy(
        results["advanced_knn_search"]["results"], 
        results["faiss_search"]["results"]
    )
    results["faiss_search"]["accuracy"] = evaluate_accuracy(
        results["faiss_search"]["results"], 
        results["advanced_linear_search"]["results"]
    )

    print(f"Advanced linear search time: {results['advanced_linear_search']['total_time']:.4f} s, avg query time: {results['advanced_linear_search']['avg_query_time'] * 1000:.6f} ms, similarity to Advanced KNN: {results['advanced_linear_search']['accuracy']:.4f}, training time: {results['advanced_linear_search']['training_time']:.4f} s")
    print(f"Advanced KNN search time: {results['advanced_knn_search']['total_time']:.4f} s, avg query time: {results['advanced_knn_search']['avg_query_time'] * 1000:.6f} ms, similarity to Faiss: {results['advanced_knn_search']['accuracy']:.4f}, training time: {results['advanced_knn_search']['training_time']:.4f} s")
    print(f"Faiss search time: {results['faiss_search']['total_time']:.4f} s, avg query time: {results['faiss_search']['avg_query_time'] * 1000:.6f} ms, similarity to Advanced Linear: {results['faiss_search']['accuracy']:.4f}, training time: {results['faiss_search']['training_time']:.4f} s")

    return results

def main():
    np.random.seed(42)
    # Add different dimensions to test
    dimensions = [128, 1024]
    metrics = ["cosine", "l2", "inner_product"]

    datasets = {
        "small": {"num_vectors": 100, "num_queries": 10, "k": 10},
        "large": {"num_vectors": 10000, "num_queries": 100, "k": 30},
    }

    large_datasets = {
        "huge": {"num_vectors": 100000, "num_queries": 200, "k": 50},
        "huge_query": {"num_vectors": 100000, "num_queries": 200, "k": 10000},
        "huge_full_query": {"num_vectors": 100000, "num_queries": 200, "k": 100000},
    }

    results = {}

    # Benchmark regular datasets with different dimensions
    for dimension in dimensions:
        results[f"dim_{dimension}"] = {}
        for dataset_name, params in datasets.items():
            vectors = generate_random_vectors(params["num_vectors"], dimension)
            queries = generate_random_vectors(params["num_queries"], dimension)
            
            results[f"dim_{dimension}"][dataset_name] = {}
            for metric in metrics:
                results[f"dim_{dimension}"][dataset_name][metric] = benchmark_dataset(
                    vectors, queries, params["k"], dataset_name, metric, dimension
                )

    # Benchmark large datasets with different dimensions
    for dimension in dimensions:
        results[f"large_dim_{dimension}"] = {}
        for dataset_name, params in large_datasets.items():
            vectors = generate_random_vectors(params["num_vectors"], dimension)
            queries = generate_random_vectors(params["num_queries"], dimension)
            
            results[f"large_dim_{dimension}"][dataset_name] = {}
            for metric in metrics:
                results[f"large_dim_{dimension}"][dataset_name][metric] = benchmark_large_dataset(
                    vectors, queries, params["k"], dataset_name, metric, dimension
                )

    # Save results
    os.makedirs('benchmarks', exist_ok=True)
    with open('benchmarks/results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
