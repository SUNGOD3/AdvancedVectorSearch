# benchmarks/run_benchmarks.py

import time
import json
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search import LinearSearch, FaissSearch, AdvancedSearch
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

def main():
    np.random.seed(42) 
    dimensions = 1024

    datasets = {
        "small": {"num_vectors": 100, "num_queries": 10, "k": 10},
        "medium": {"num_vectors": 1000, "num_queries": 100, "k": 20},
        "large": {"num_vectors": 10000, "num_queries": 100, "k": 30},
    }

    large_datasets = {
        "extra_large": {"num_vectors": 50000, "num_queries": 200, "k": 50},
        "huge": {"num_vectors": 100000, "num_queries": 200, "k": 50},
        "huge_query": {"num_vectors": 100000, "num_queries": 200, "k": 5000},
    }

    results = {}

    # Benchmark small datasets
    for dataset_name, params in datasets.items():
        num_vectors = params["num_vectors"]
        num_queries = params["num_queries"]
        k = params["k"]

        print(f"Running benchmark for {dataset_name} dataset with {num_queries} queries and k={k}...")
        vectors = generate_random_vectors(num_vectors, dimensions)
        queries = generate_random_vectors(num_queries, dimensions)

        linear_search = LinearSearch(vectors)
        advanced_search = AdvancedSearch(vectors)
        faiss_search = FaissSearch(vectors)

        linear_results, linear_time = run_benchmark(linear_search, vectors, queries, k)
        advanced_results, advanced_time = run_benchmark(advanced_search, vectors, queries, k)
        faiss_results, faiss_time = run_benchmark(faiss_search, vectors, queries, k)

        advanced_accuracy = evaluate_accuracy(linear_results, advanced_results)
        faiss_accuracy = evaluate_accuracy(linear_results, faiss_results)

        results[dataset_name] = {
            "linear_search": {
                "time": linear_time
            },
            "advanced_search": {
                "time": advanced_time,
                "accuracy": advanced_accuracy
            },
            "faiss_search": {
                "time": faiss_time,
                "accuracy": faiss_accuracy
            }
        }

        print(f"Linear search time: {linear_time:.4f} s")
        print(f"Advanced search time: {advanced_time:.4f} s, accuracy: {advanced_accuracy:.4f}")
        print(f"Faiss search time: {faiss_time:.4f} s, accuracy: {faiss_accuracy:.4f}")

    # Benchmark large datasets
    for dataset_name, params in large_datasets.items():
        print(f"\nRunning benchmark for {dataset_name} dataset with {params['num_queries']} queries and k={params['k']}...")
        vectors = generate_random_vectors(params["num_vectors"], dimensions)
        queries = generate_random_vectors(params["num_queries"], dimensions)

        advanced_search = AdvancedSearch(vectors)
        faiss_search = FaissSearch(vectors)

        advanced_results, advanced_time = run_benchmark(advanced_search, vectors, queries, params["k"])
        faiss_results, faiss_time = run_benchmark(faiss_search, vectors, queries, params["k"])

        results[dataset_name] = {
            "advanced_search": {"time": advanced_time},
            "faiss_search": {"time": faiss_time}
        }

        print(f"Advanced search time: {advanced_time:.4f} s")
        print(f"Faiss search time: {faiss_time:.4f} s")

    os.makedirs('benchmarks', exist_ok=True)
    with open('benchmarks/results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()