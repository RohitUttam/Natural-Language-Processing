import numpy as np
import pandas as pd
from collections import defaultdict

class Autocompleter():
    
    @staticmethod
    def ngram_count_matrix(n: int, corpus: list):
        """
        Creates the trigram count matrix from the input corpus in a single pass through the corpus.
        
        Parameters
        ----------
            n: int
                size of n-gram
            corpus: str
                Pre-processed and tokenized corpus. 
        
        Returns
        -------
            ngrams: list of all ngram prefixes, row index
            vocabulary: list of all found words, the column index
            count_matrix: probabilities of the n-gram/word combinations
        """
        ngrams = []
        vocabulary = []
        count_matrix_dict = defaultdict(dict)
        
        # sliding window
        for i in range(len(corpus) - n):
            
            # n = n-1 + last word:
            ngram = tuple(corpus[i : i + n + 1])
            
            # without last word:
            n_1_gram = ngram[0 : -1]
            
            if not n_1_gram in ngrams:
                ngrams.append(n_1_gram)        
            
            last_word = ngram[-1]
            if not last_word in vocabulary:
                vocabulary.append(last_word)
            
            if (n_1_gram,last_word) not in count_matrix_dict:
                count_matrix_dict[n_1_gram,last_word] = 0
                
            count_matrix_dict[n_1_gram,last_word] += 1
        
        # convert the count_matrix to np.array to fill in the blanks
        count_matrix = np.zeros((len(ngrams), len(vocabulary)))
        for ngram_key, ngram_count in count_matrix_dict.items():
            count_matrix[ngrams.index(ngram_key[0]), \
                        vocabulary.index(ngram_key[1])] = ngram_count
        
        # Row normalize:
        count_matrix =(count_matrix.T/count_matrix.sum(axis=1)).T

        count_matrix = pd.DataFrame(count_matrix, index=ngrams, columns=vocabulary)
        
        return ngrams, vocabulary, count_matrix