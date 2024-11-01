import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search import LinearSearch, FaissSearch, AdvancedSearch
from src.data_generator import generate_random_vectors

def read_fvecs(filename: str) -> np.ndarray:
    """
    Read SIFT descriptors from .fvecs file format.
    .fvecs format: Each vector is stored as <dim><float32>^dim where dim is int32
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_data = f.read(4)  # Read dimension (int32)
            if not dim_data:  # EOF
                break
            
            dim = np.frombuffer(dim_data, dtype=np.int32)[0]
            vector_data = f.read(dim * 4)  # Read dim * float32
            if not vector_data:  # EOF
                break
            
            vector = np.frombuffer(vector_data, dtype=np.float32)
            vectors.append(vector)
    
    return np.array(vectors)

def read_ivecs(filename: str) -> np.ndarray:
    """
    Read ground truth from .ivecs file format.
    .ivecs format: Each vector is stored as <dim><int32>^dim where dim is int32
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_data = f.read(4)  # Read dimension (int32)
            if not dim_data:  # EOF
                break
            
            dim = np.frombuffer(dim_data, dtype=np.int32)[0]
            vector_data = f.read(dim * 4)  # Read dim * int32
            if not vector_data:  # EOF
                break
            
            vector = np.frombuffer(vector_data, dtype=np.int32)
            vectors.append(vector)
    
    return np.array(vectors)

def calculate_recall_with_debug(predicted: List[int], actual: List[int], k: int) -> float:
    """
    Calculate recall@k metric with detailed debugging information.
    """
    predicted_set = set(predicted[:k])
    actual_set = set(actual[:k])
    intersection = predicted_set.intersection(actual_set)
    recall = len(intersection) / k
    
    # Debug information
    missing_items = actual_set - predicted_set
    extra_items = predicted_set - actual_set
    
    return {
        'recall': recall,
        'predicted': sorted(list(predicted_set)),
        'actual': sorted(list(actual_set)),
        'intersection': sorted(list(intersection)),
        'missing': sorted(list(missing_items)),
        'extra': sorted(list(extra_items))
    }

class BenchmarkRunner:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # Load datasets
        print("Loading datasets...")
        self.base_vectors = read_fvecs(str(self.data_dir / "siftsmall_base.fvecs"))
        self.query_vectors = read_fvecs(str(self.data_dir / "siftsmall_query.fvecs"))
        self.ground_truth = read_ivecs(str(self.data_dir / "siftsmall_groundtruth.ivecs"))
        
        print(f"Loaded {len(self.base_vectors)} base vectors with dimension {self.base_vectors.shape[1]}")
        print(f"Loaded {len(self.query_vectors)} query vectors with dimension {self.query_vectors.shape[1]}")
        print(f"Loaded {len(self.ground_truth)} ground truth vectors")
        
    def calculate_recall(self, predicted: List[int], actual: List[int], k: int) -> float:
        """Calculate recall@k metric."""
        predicted_set = set(predicted[:k])
        actual_set = set(actual[:k])
        return len(predicted_set.intersection(actual_set)) / k

    def run_benchmark(self, search_class, k: int = 100, num_queries: int = None) -> dict:
        """
        Run benchmark for a given search implementation.
        
        Args:
            search_class: Class implementing the search interface
            k: Number of nearest neighbors to retrieve
            num_queries: Number of queries to test (None for all)
        """
        if num_queries is None:
            num_queries = len(self.query_vectors)
        
        # Initialize search index
        start_time = time.time()
        search_instance = search_class(self.base_vectors)
        build_time = time.time() - start_time
        
        recalls = []
        query_times = []
        
        # Run queries
        for i in range(num_queries):
            query = self.query_vectors[i]
            ground_truth = self.ground_truth[i]
            
            # Measure search time
            start_time = time.time()
            results = search_instance.search(query, k)
            query_time = time.time() - start_time
            
            # Calculate recall
            recall = self.calculate_recall(results, ground_truth, k)
            
            recalls.append(recall)
            query_times.append(query_time)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_queries} queries")
        
        # Compile results
        results = {
            'avg_recall': np.mean(recalls),
            'avg_query_time': np.mean(query_times),
            'index_build_time': build_time,
            'recalls': recalls,
            'query_times': query_times
        }
        
        return results

    def plot_results(self, results_dict: dict):
        """Plot benchmark results."""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Average Performance Comparison
        plt.subplot(1, 2, 1)
        methods = list(results_dict.keys())
        recalls = [results_dict[m]['avg_recall'] for m in methods]
        query_times = [results_dict[m]['avg_query_time'] * 1000 for m in methods]  # Convert to ms
        
        x = range(len(methods))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], recalls, width, label='Recall@100')
        plt.bar([i + width/2 for i in x], query_times, width, label='Query Time (ms)')
        
        plt.xlabel('Method')
        plt.ylabel('Score')
        plt.title('Performance Comparison')
        plt.xticks(x, methods)
        plt.legend()
        
        # Plot 2: Query Time Distribution
        plt.subplot(1, 2, 2)
        for method in methods:
            times = np.array(results_dict[method]['query_times']) * 1000  # Convert to ms
            plt.hist(times, alpha=0.5, label=method, bins=30)
        
        plt.xlabel('Query Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Query Time Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize benchmark runner
    data_path = "data/sift10k/siftsmall"
    print(f"Loading data from: {data_path}")
    benchmark = BenchmarkRunner(data_path)
    
    # Define search methods to benchmark
    search_methods = {
        'Linear': LinearSearch,
        'Advanced': AdvancedSearch,
        'Faiss': FaissSearch
    }
    
    # Run benchmarks
    results = {}
    for name, method in search_methods.items():
        print(f"\nRunning benchmark for {name}...")
        results[name] = benchmark.run_benchmark(method, k=100, num_queries=100)
        print(f"Average Recall@100: {results[name]['avg_recall']:.3f}")
        print(f"Average Query Time: {results[name]['avg_query_time']*1000:.2f}ms")
        print(f"Index Build Time: {results[name]['index_build_time']:.2f}s")
    
    # Plot results
    benchmark.plot_results(results)

if __name__ == "__main__":
    main()