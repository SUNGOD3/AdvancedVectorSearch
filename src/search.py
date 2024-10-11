# src/search.py

import numpy as np
from scipy.spatial.distance import cosine
import faiss

def calculate_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

class LinearSearch:
    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, query, k):
        """
        执行线性搜索来找到最近的 k 个向量。
        
        :param query: 查询向量
        :param k: 要返回的最近邻数量
        :return: 最相似向量的索引列表
        """
        distances = [calculate_similarity(query, vector) for vector in self.vectors]
        return np.argsort(distances)[::-1][:k]

class AdvancedSearch:
    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, query, k):
        """
        执行"高级"搜索（实际上只是线性搜索的一个包装器）。
        在实际实现中，这里应该包含更高级的搜索算法。
        
        :param query: 查询向量
        :param k: 要返回的最近邻数量
        :return: 最相似向量的索引列表
        """
        # 这里我们只是调用 LinearSearch 来模拟一个"高级"搜索
        linear_search = LinearSearch(self.vectors)
        return linear_search.search(query, k)

class FaissSearch:
    def __init__(self, vectors):
        self.vectors = np.array(vectors).astype('float32')
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])  # 使用內積（cosine similarity）
        faiss.normalize_L2(self.vectors)  # 正規化向量以使用 cosine similarity
        self.index.add(self.vectors)

    def search(self, query, k):
        query = np.array(query).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query)  # 正規化查詢向量
        distances, indices = self.index.search(query, k)
        return indices[0]