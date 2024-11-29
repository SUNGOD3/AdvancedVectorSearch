# src/search.py
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from advanced_search_cpp import AdvancedLinearSearch as CppAdvancedLinearSearch
from advanced_search_cpp import AdvancedKNNSearch as CppAdvancedKNNSearch
import hnswlib
import faiss
from enum import Enum

class DistanceMetric(Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"

class AdvancedLinearSearch():
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

class AdvancedKNNSearch():
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


class AdvancedHNSWSearch:
    def __init__(self, vectors, metric="cosine", ef_construction=150, M=16):
        """
        Initialize the AdvancedHNSWSearch instance using HNSW algorithm.
        
        :param vectors: A 2D numpy array of vectors to search through
        :param metric: Distance metric to use ("cosine" or "l2")
        :param ef_construction: Depth of layer construction (higher = more accurate but slower)
        :param M: Maximum number of connections per element (higher = more accurate but slower)
        """
        # Ensure input is a numpy array
        vectors = np.asarray(vectors, dtype=np.float32)
        
        # Get vector dimension
        dim = vectors.shape[1]
        
        # Normalize vectors if using cosine similarity or inner product
        if metric in ["cosine", "inner_product"]:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms!=0)
        
        # Create HNSW index with more careful configuration
        if metric == 'l2':
            space = 'l2'
        elif metric in ['cosine', 'inner_product']:
            space = 'ip'  # inner product for both cosine and inner_product
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Create index with optimized parameters
        self.index = hnswlib.Index(space=space, dim=dim)
        
        # More aggressive index construction parameters
        self.index.init_index(max_elements=vectors.shape[0], 
                               ef_construction=ef_construction,  # Increased from 200 to 400
                               M=M)  # Increased from 16 to 64
        
        # Add all vectors to the index
        self.index.add_items(vectors)
        
        # Optimize search parameters
        # Significantly increase ef parameter for more thorough search
        self.index.set_ef(max(50, vectors.shape[0] // 10))  # Dynamic ef based on dataset size
        
        # Store metric for potential future use
        self.metric = metric

    def search(self, query, k):
        """
        Perform HNSW search using hnswlib with improved accuracy.
        
        :param query: Query vector
        :param k: Number of nearest neighbors to return
        :return: List of indices of the most similar vectors
        """
        # Ensure query is a numpy array with float32 type
        query = np.asarray(query, dtype=np.float32)
        
        # Normalize query if using cosine or inner product
        if self.metric in ["cosine", "inner_product"]:
            norm = np.linalg.norm(query)
            query = query / norm if norm != 0 else query

        # Perform the search
        labels, _ = self.index.knn_query(query, k=k)
        
        # Return the list of indices
        return labels[0].tolist()

    def add(self, vector):
        """
        Add a new vector to the search index.
        
        :param vector: Vector to add
        """
        # Ensure vector is a numpy array
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        
        # Normalize if using cosine or inner product
        if self.metric in ["cosine", "inner_product"]:
            norm = np.linalg.norm(vector)
            vector = vector / norm if norm != 0 else vector
        
        # Add the vector to the index
        self.index.add_items(vector)


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
        if self.metric in ["cosine", "inner_product"]:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.vectors = self.vectors / norms
    
    def _compute_distances(self, query):
        if self.metric == "cosine":
            # Normalize query for cosine similarity
            query = query / np.linalg.norm(query)
            # Compute cosine distances
            distances = np.array([cosine(query, v) for v in self.vectors])
        elif self.metric == "inner_product":
            # Normalize query for inner product
            query = query / np.linalg.norm(query)
            # Compute negative inner product (for consistent sorting)
            distances = np.array([-np.dot(query, v) for v in self.vectors])
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
        elif self.metric == "inner_product":
            # For inner product, normalize vectors and use IndexFlatIP
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
        
        if self.metric in ["cosine", "inner_product"]:
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
        
        if self.metric in ["cosine", "inner_product"]:
            faiss.normalize_L2(vectors)
            
        self.index.add(vectors)