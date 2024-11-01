# src/search.py

import numpy as np
from scipy.spatial.distance import cosine
from advanced_search_cpp import AdvancedSearch as CppAdvancedSearch
import faiss

class AdvancedSearch:
    def __init__(self, vectors):
        """
        Initialize the AdvancedSearch instance.
        
        :param vectors: A 2D numpy array of vectors to search through
        """
        self.cpp_search = CppAdvancedSearch(vectors)

    def search(self, query, k):
        """
        Perform an "advanced" search using the C++ extension.
        
        :param query: Query vector
        :param k: Number of nearest neighbors to return
        :return: List of indices of the most similar vectors
        """
        # Ensure query is a numpy array
        query = np.asarray(query, dtype=np.float64)
        
        # Perform the search using the C++ extension
        result = self.cpp_search.search(query, k)
        
        # Convert the result to a Python list and return
        return result.tolist()


class LinearSearch:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

    def search(self, query, k):
        query = np.array(query)
        distances = np.array([cosine(query, v) for v in self.vectors])
        top_k_indices = np.argsort(distances)[:k]
        similarities = 1 - distances[top_k_indices]  # Convert distance to similarity
        return top_k_indices.tolist()#, similarities.tolist()

    def add(self, vector):
        vector = np.array(vector).reshape(1, -1)
        self.vectors = np.vstack((self.vectors, vector))

# faiss.normalize_L2 -> faiss/utils/distances.cpp
class FaissSearch:
    def __init__(self, vectors):
        self.dimension = vectors.shape[1]
        self.vectors = vectors

        # Normalize vectors
        faiss.normalize_L2(self.vectors)

        # Use IndexFlatIP to compute inner product (for normalized vectors, this is equivalent to cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.vectors)

    def search(self, query, k):
        query = np.array(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        similarities, indices = self.index.search(query, k)
        
        # Faiss returns similarities as cosine similarity, no conversion needed
        return indices[0].tolist()#, similarities[0].tolist()

    def add(self, vectors):
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

