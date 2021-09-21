import numpy as np

class FindSimilarText():

    def __init__(self):
        pass 
    
    @staticmethod
    def min_hamming_distance(source, target, ins_cost = 1, del_cost = 1, rep_cost = 2):
        m,n = len(source),len(target) 
        D = np.zeros((m+1, n+1), dtype=int) 
        for row in range(1,m+1): 
            D[row,0] =  D[row-1,0]+del_cost
        for col in range(1,n+1):
            D[0,col] = D[0,col-1]+ins_cost
        for row in range(1,m+1): 
            for col in range(1,n+1):
                r_cost = rep_cost
                if source[row-1]==target[col-1]:
                    r_cost = 0
                replacement_cost = D[row-1,col-1]+r_cost
                deletion_cost = D[row-1,col]+del_cost
                insertion_cost = D[row,col-1]+ins_cost
                D[row,col] = np.min(np.array([replacement_cost,deletion_cost,insertion_cost]))
            
        minimum_distance = D[m,n]
        
        return minimum_distance, D

    @classmethod
    def find_matches(cls, name, candidates, k=1, max_distance=100):
        lista_names=[None]
        list_distances=[max_distance]
        for candidate in candidates:
            d,_ = cls.min_hamming_distance(name,candidate)
            lista_names.append(candidate)
            list_distances.append(d)
        sorted_ids = np.argsort(np.array(list_distances))
        k_idx = sorted_ids[:k]
        return np.array(lista_names)[k_idx]

    @staticmethod
    def cosine_similarity(u,v):
        def norm(u):
            return np.sqrt(np.sum(u**2))
        return np.dot(u,v)/(norm(u)*norm(v))

    @classmethod
    def k_nearest_neighbor(cls, vector, candidates, k=1):
        """
        Parameters
        ----------
        vector: np.array
            the vector you are going find the nearest neighbor for
        candidates: np.array
            a set of vectors where we will find the neighbors
        k: int
            top k nearest neighbors to find

        Result
        ------
        k_idx: int
            the indices of the top k closest vectors in sorted form
        """
        similarity_l = []

        for row in candidates:
            cos_similarity = cls.cosine_similarity(vector,row)
            similarity_l.append(cos_similarity)
        
        sorted_ids = np.argsort(similarity_l)
        k_idx = sorted_ids[-k:]
        return k_idx