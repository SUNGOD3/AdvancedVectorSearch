# src/search.py

import numpy as np
from scipy.spatial.distance import cosine
import faiss


class AdvancedSearch:
    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, query, k):
        """
        Perform an "advanced" search (actually just a wrapper for linear search).
        In a real implementation, this should contain a more advanced search algorithm.
        
        :param query: Query vector
        :param k: Number of nearest neighbors to return
        :return: List of indices of the most similar vectors
        """
        linear_search = LinearSearch(self.vectors)
        return linear_search.search(query, k)

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

