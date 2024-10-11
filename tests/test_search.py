
import unittest
import numpy as np
from src.search import LinearSearch, AdvancedSearch, FaissSearch
from src.data_generator import generate_random_vectors

class TestSearch(unittest.TestCase):
    def setUp(self):
        self.dim = 128
        self.num_vectors = 1000
        self.num_queries = 10
        self.k = 5
        self.vectors = generate_random_vectors(self.num_vectors, self.dim)
        self.queries = generate_random_vectors(self.num_queries, self.dim)

    def test_linear_search(self):
        linear_search = LinearSearch(self.vectors)
        results = linear_search.search(self.queries[0], self.k)
        self.assertEqual(len(results), self.k)
        # Add more assertions here
        for result in results:
            self.assertIn(result, range(self.num_vectors))

    def test_advanced_search(self):
        advanced_search = AdvancedSearch(self.vectors)
        results = advanced_search.search(self.queries[0], self.k)
        self.assertEqual(len(results), self.k)
        # Add more assertions here
        for result in results:
            self.assertIn(result, range(self.num_vectors))

    def test_faiss_search(self):
        faiss_search = FaissSearch(self.vectors)
        results = faiss_search.search(self.queries[0], self.k)
        self.assertEqual(len(results), self.k)
        # Add more assertions here
        for result in results:
            self.assertIn(result, range(self.num_vectors))