# src/search.py

import numpy as np

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
        distances = np.linalg.norm(self.vectors - query, axis=1)
        return np.argsort(distances)[:k]

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