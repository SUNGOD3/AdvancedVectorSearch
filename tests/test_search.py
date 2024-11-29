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

    def test_linear_search(self):
        linear_search = LinearSearch(self.vectors)
        results = linear_search.search(self.queries[0], self.k)
        self.assertEqual(len(results), self.k)
        for result in results:
            self.assertIn(result, range(self.num_vectors))

    def test_advanced_linear_search(self):
        advanced_linear = AdvancedLinearSearch(self.vectors)
        results = advanced_linear.search(self.queries[0], self.k)
        self.assertEqual(len(results), self.k)
        for result in results:
            self.assertIn(result, range(self.num_vectors))

    def test_advanced_knn_search(self):
        advanced_knn = AdvancedKNNSearch(self.vectors)
        results = advanced_knn.search(self.queries[0], self.k)
        self.assertEqual(len(results), self.k)
        for result in results:
            self.assertIn(result, range(self.num_vectors))

    def test_faiss_search(self):
        faiss_search = FaissSearch(self.vectors)
        results = faiss_search.search(self.queries[0], self.k)
        self.assertEqual(len(results), self.k)
        for result in results:
            self.assertIn(result, range(self.num_vectors))

    def test_results_consistency(self):
        """Test that all search methods return similar results"""
        query = self.queries[0]
        
        # Get results from all methods
        linear_results = set(LinearSearch(self.vectors).search(query, self.k))
        advanced_linear_results = set(AdvancedLinearSearch(self.vectors).search(query, self.k))
        advanced_knn_results = set(AdvancedKNNSearch(self.vectors).search(query, self.k))
        faiss_results = set(FaissSearch(self.vectors).search(query, self.k))
        
        # Calculate Jaccard similarity between results
        def jaccard_similarity(set1, set2):
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0
        
        # Compare results with a tolerance
        tolerance = 0.0 # no tolerance
        
        self.assertGreater(jaccard_similarity(linear_results, advanced_linear_results), tolerance)
        self.assertGreater(jaccard_similarity(linear_results, advanced_knn_results), tolerance)
        self.assertGreater(jaccard_similarity(linear_results, faiss_results), tolerance)
        self.assertGreater(jaccard_similarity(advanced_linear_results, advanced_knn_results), tolerance)
        self.assertGreater(jaccard_similarity(advanced_linear_results, faiss_results), tolerance)
        self.assertGreater(jaccard_similarity(advanced_knn_results, faiss_results), tolerance)

    def test_edge_cases(self):
        """Test edge cases for all search methods"""
        # Test with k = 1
        k = 1
        self.assertEqual(len(LinearSearch(self.vectors).search(self.queries[0], k)), k)
        self.assertEqual(len(AdvancedLinearSearch(self.vectors).search(self.queries[0], k)), k)
        self.assertEqual(len(AdvancedKNNSearch(self.vectors).search(self.queries[0], k)), k)
        self.assertEqual(len(FaissSearch(self.vectors).search(self.queries[0], k)), k)

        # Test with k = num_vectors
        k = self.num_vectors
        self.assertEqual(len(LinearSearch(self.vectors).search(self.queries[0], k)), k)
        self.assertEqual(len(AdvancedLinearSearch(self.vectors).search(self.queries[0], k)), k)
        self.assertEqual(len(AdvancedKNNSearch(self.vectors).search(self.queries[0], k)), k)
        self.assertEqual(len(FaissSearch(self.vectors).search(self.queries[0], k)), k)

    def test_input_validation(self):
        """Test input validation for all search methods"""
        invalid_query = np.random.rand(self.dim + 1).astype(np.float32)  # Wrong dimension
        
        with self.assertRaises(Exception):
            LinearSearch(self.vectors).search(invalid_query, self.k)
        with self.assertRaises(Exception):
            AdvancedLinearSearch(self.vectors).search(invalid_query, self.k)
        with self.assertRaises(Exception):
            AdvancedKNNSearch(self.vectors).search(invalid_query, self.k)
        with self.assertRaises(Exception):
            FaissSearch(self.vectors).search(invalid_query, self.k)

if __name__ == '__main__':
    unittest.main()