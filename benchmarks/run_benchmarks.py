import time
import json
import numpy as np
from src.search import LinearSearch, AdvancedSearch
from src.data_generator import generate_random_vectors

def run_benchmark(search_method, vectors, queries, k):
    start_time = time.time()
    for query in queries:
        search_method.search(query, k)
    end_time = time.time()
    return end_time - start_time

def main():
    np.random.seed(2486)  # 固定随机种子
    dimensions = 128
    num_queries = 100
    k = 10

    datasets = {
        "small": 100,
        "medium": 1000000,
        "large": 1000000000
    }

    results = {}

    for dataset_name, num_vectors in datasets.items():
        vectors = generate_random_vectors(num_vectors, dimensions)
        queries = generate_random_vectors(num_queries, dimensions)

        linear_search = LinearSearch(vectors)
        advanced_search = AdvancedSearch(vectors)

        results[dataset_name] = {
            "linear_search": run_benchmark(linear_search, vectors, queries, k),
            "advanced_search": run_benchmark(advanced_search, vectors, queries, k)
        }

    with open('benchmarks/results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()