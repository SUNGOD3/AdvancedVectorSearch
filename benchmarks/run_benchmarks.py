# benchmarks/run_benchmarks.py

import time
import json
import numpy as np
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search import LinearSearch, AdvancedSearch
from src.data_generator import generate_random_vectors

def run_benchmark(search_method, vectors, queries, k):
    start_time = time.time()
    for query in queries:
        search_method.search(query, k)
    end_time = time.time()
    return end_time - start_time

def main():
    np.random.seed(42)  # 固定随机种子
    dimensions = 128
    num_queries = 100
    k = 10

    datasets = {
        "small": 100,
        "medium": 1000,  # 减小数据集大小以加快测试
        "large": 10000   # 减小数据集大小以加快测试
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