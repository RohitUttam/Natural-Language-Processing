
import numpy as np

class EmbeddingTranslator():
    
    def __init__(self, embedding_language1, embedding_language2, R):
        self.embedding_language1 = embedding_language1
        self.embedding_language2 = embedding_language2
        self.R = R

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

    def translate(self, word: str, how: str = '1to2'):
        """Translate word through 1-nearest-neighbour.
        
        Parameters
        ----------
        word: str
            word to translate between embeddings
        how: str
            '1to2' or '2to1', direction of translation
        
        Result
        ------
        translation: str
            word translated
        """
        if (how=='1to2') and (word in self.embedding_language1):
            word_vector = self.embedding_language1[word]
            word_vector_translated = np.matmul(word_vector,self.R)
            candidates = np.array(list(self.embedding_language2.values()))
            candidates_text = list(self.embedding_language2.keys())
            translation_idx = self.k_nearest_neighbor(word_vector_translated,candidates,k=1)
            translation = candidates_text[int(translation_idx)]
        elif (how=='2to1') and (word in self.embedding_language2):
            word_vector = self.embedding_language2[word]
            word_vector_translated = np.matmul(word_vector,self.R)
            candidates = self.embedding_language1
            candidates_text = list(self.embedding_language2.keys())
            translation_idx = self.k_nearest_neighbor(word_vector_translated,candidates,k=1)
            translation = candidates_text[int(translation_idx)]
        else:
            translation =  None
        return translation
        