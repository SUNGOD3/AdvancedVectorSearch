# src/data_generator.py

import numpy as np

def generate_random_vectors(num_vectors, dimensions):
    """
    生成随机向量。
    
    :param num_vectors: 要生成的向量数量
    :param dimensions: 每个向量的维度
    :return: 形状为 (num_vectors, dimensions) 的 numpy 数组
    """
    return np.random.rand(num_vectors, dimensions).astype(np.float32)