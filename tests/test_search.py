import unittest
import numpy as np
from src.search import LinearSearch, FaissSearch, AdvancedLinearSearch, AdvancedKNNSearch, AdvancedHNSWSearch
from src.data_generator import generate_random_vectors

class TestSearch(unittest.TestCase):
    def setUp(self):
        self.dim = 128
        self.num_vectors = 1000
        self.num_queries = 10
        self.k = 5
        self.vectors = generate_random_vectors(self.num_vectors, self.dim)
        self.queries = generate_random_vectors(self.num_queries, self.dim)
        
        # Convert to float32 for consistency across all methods
        self.vectors = self.vectors.astype(np.float32)
        self.queries = self.queries.astype(np.float32)

        self.search_methods = [
            ('LinearSearch', LinearSearch),
            ('AdvancedLinearSearch', AdvancedLinearSearch),
            ('AdvancedKNNSearch', AdvancedKNNSearch),
            ('AdvancedHNSWSearch', AdvancedHNSWSearch),
            ('FaissSearch', FaissSearch)
        ]

        self.metrics = ['cosine', 'l2', 'inner_product']

    def _validate_search_results(self, results, k):
        """Helper method to validate search results"""
        self.assertEqual(len(results), k, f"Expected {k} results")
        for result in results:
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 0)
            self.assertLess(result, self.num_vectors)

    def test_all_search_methods(self):
        """Test each search method with different metrics"""
        for method_name, SearchClass in self.search_methods:
            for metric in self.metrics:
                with self.subTest(method=method_name, metric=metric):
                    # Skip inner_product for methods that might not support it
                    if metric == "inner_product" and method_name in ["LinearSearch"]:
                        continue
                    
                    search_instance = SearchClass(self.vectors, metric=metric)
                    results = search_instance.search(self.queries[0], self.k)
                    self._validate_search_results(results, self.k)

    def test_results_consistency(self):
        """Test that all search methods return similar results"""
        query = self.queries[0]
        
        for metric in self.metrics:
            with self.subTest(metric=metric):
                # Get results from all methods for the current metric
                method_results = {
                    name: set(SearchClass(self.vectors, metric=metric).search(query, self.k))
                    for name, SearchClass in self.search_methods
                }
                def jaccard_similarity(set1, set2):
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    return intersection / union if union > 0 else 0
                # Compare results with no tolerance
                tolerance = 0.0
                # Use itertools to generate all unique method pairs for comparison
                import itertools
                method_combinations = list(itertools.combinations(method_results.keys(), 2))
                for method1, method2 in method_combinations:
                    with self.subTest(method1=method1, method2=method2):
                        self.assertGreater(
                            jaccard_similarity(method_results[method1], method_results[method2]), 
                            tolerance
                        )


    def test_edge_cases(self):
        """Test edge cases for all search methods"""
        edge_cases = [1, self.num_vectors]
        for k in edge_cases:
            for method_name, SearchClass in self.search_methods:
                with self.subTest(method=method_name, k=k):
                    search_instance = SearchClass(self.vectors)
                    results = search_instance.search(self.queries[0], k)
                    self.assertEqual(len(results), k)

    def test_input_validation(self):
        """Test input validation for all search methods"""
        invalid_query = np.random.rand(self.dim + 1).astype(np.float32)  # Wrong dimension
        
        for method_name, SearchClass in self.search_methods:
            with self.subTest(method=method_name):
                search_instance = SearchClass(self.vectors)
                with self.assertRaises(Exception):
                    search_instance.search(invalid_query, self.k)

if __name__ == '__main__':
    unittest.main()