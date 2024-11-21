# src/search.py
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from advanced_search_cpp import AdvancedLinearSearch as CppAdvancedLinearSearch
from advanced_search_cpp import AdvancedKNNSearch as CppAdvancedKNNSearch
import faiss
from enum import Enum

class DistanceMetric(Enum):
    COSINE = "cosine"
    L2 = "l2"

class AdvancedSearchBase:
    def __init__(self, metric=DistanceMetric.COSINE):
        self.metric = metric
    
    def search(self, query, k):
        """
        Base search method to be implemented by subclasses
        """
        raise NotImplementedError

class AdvancedLinearSearch(AdvancedSearchBase):
    def __init__(self, vectors, metric="cosine"):
        """
        Initialize the AdvancedLinearSearch instance.
        
        :param vectors: A 2D numpy array of vectors to search through
        :param metric: Distance metric to use ("cosine" or "l2")
        """
        self.cpp_search = CppAdvancedLinearSearch(vectors, metric)
    
    def search(self, query, k):
        """
        Perform a linear search using the C++ extension.
        
        :param query: Query vector
        :param k: Number of nearest neighbors to return
        :return: List of indices of the most similar vectors
        """
        query = np.asarray(query, dtype=np.float32)
        return self.cpp_search.search(query, k).tolist()

class AdvancedKNNSearch(AdvancedSearchBase):
    def __init__(self, vectors, metric="cosine"):
        """
        Initialize the AdvancedKNNSearch instance.
        
        :param vectors: A 2D numpy array of vectors to search through
        :param metric: Distance metric to use ("cosine" or "l2")
        """
        self.cpp_search = CppAdvancedKNNSearch(vectors, metric)

    def search(self, query, k):
        """
        Perform KNN search using the C++ extension.
        
        :param query: Query vector
        :param k: Number of nearest neighbors to return
        :return: List of indices of the most similar vectors
        """
        query = np.asarray(query, dtype=np.float32)
        return self.cpp_search.search(query, k).tolist()


class LinearSearch:
    def __init__(self, vectors, metric="cosine"):
        """
        Initialize the LinearSearch instance.
        
        :param vectors: A 2D numpy array of vectors to search through
        :param metric: Distance metric to use ("cosine" or "l2")
        """
        self.vectors = np.array(vectors)
        self.metric = metric
        
        # Normalize vectors if using cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.vectors = self.vectors / norms
    
    def _compute_distances(self, query):
        if self.metric == "cosine":
            # Normalize query for cosine similarity
            query = query / np.linalg.norm(query)
            # Compute cosine distances
            distances = np.array([cosine(query, v) for v in self.vectors])
        else:  # L2
            # Compute L2 distances
            distances = np.array([euclidean(query, v) for v in self.vectors])
        return distances
    
    def search(self, query, k):
        """
        Perform linear search through vectors.
        
        :param query: Query vector
        :param k: Number of nearest neighbors to return
        :return: List of indices of the most similar vectors
        """
        query = np.array(query)
        distances = self._compute_distances(query)
        top_k_indices = np.argsort(distances)[:k]
        return top_k_indices.tolist()
    
    def add(self, vector):
        """
        Add a new vector to the search index.
        
        :param vector: Vector to add
        """
        vector = np.array(vector).reshape(1, -1)
        if self.metric == "cosine":
            vector = vector / np.linalg.norm(vector)
        self.vectors = np.vstack((self.vectors, vector))

class FaissSearch:
    def __init__(self, vectors, metric="cosine"):
        """
        Initialize the FaissSearch instance.
        
        :param vectors: A 2D numpy array of vectors to search through
        :param metric: Distance metric to use ("cosine" or "l2")
        """
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.metric = metric
        
        if self.metric == "cosine":
            # For cosine similarity, normalize vectors and use IndexFlatIP
            faiss.normalize_L2(self.vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        else:  # L2
            # For L2 distance, use IndexFlatL2
            self.index = faiss.IndexFlatL2(self.dimension)
            
        self.index.add(self.vectors)
    
    def search(self, query, k):
        """
        Perform search using Faiss index.
        
        :param query: Query vector
        :param k: Number of nearest neighbors to return
        :return: List of indices of the most similar vectors
        """
        query = np.array(query).reshape(1, -1).astype('float32')
        
        if self.metric == "cosine":
            faiss.normalize_L2(query)
            
        distances, indices = self.index.search(query, k)
        return indices[0].tolist()
    
    def add(self, vectors):
        """
        Add new vectors to the search index.
        
        :param vectors: Vectors to add
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        vectors = vectors.astype('float32')
        
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
            
        self.index.add(vectors)