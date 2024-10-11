# src/data_generator.py

import numpy as np

def generate_random_vectors(num_vectors, dimensions):
    """
    Generate random vectors.
    
    :param num_vectors: Number of vectors to generate
    :param dimensions: Dimensions of each vector
    :return: A numpy array of shape (num_vectors, dimensions)
    """
    return np.random.rand(num_vectors, dimensions).astype(np.float32)