import re
import string
import numpy as np
from .preprocessor import TextPreprocessor

class SentimentAnalyzer():

    def __init__(self, language='english'):
        self.preprocessor = TextPreprocessor(language)
        self.language = language
        self.frequencies = None
        self.all_words = None
        self.dict_negative = None
        self.dict_positive = None

    def build_frequencies(self, list_texts: list, ys:list, language: str = None):
        """Build frequencies.

        Parameters
        ----------
            tweets: list of tweets
            ys: binary labels (0: negative, 1:positive)

        Result
        ------
            freqs: mapping (word, sentiment) to frequency
        """

        if language is None:
            language = self.language

        yslist = np.squeeze(ys).tolist()

        freqs = {}
        all_words = set()
        for y, text in zip(yslist, list_texts):
            processed_text = self.preprocessor.process_text(text,language=language)
            for word in processed_text:
                all_words.add(word)
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1

        self.frequencies = freqs
        self.all_words = all_words
        return freqs

    def get_word_frequencies(self, list_of_words, dict_freqs=None):
        if dict_freqs is None:
            dict_freqs = self.frequencies
        data = {}
        for word in list_of_words:            
            pos = dict_freqs.get((word, 1),0)
            neg = dict_freqs.get((word, 0),0)
            data[word] =  [pos, neg]
        return data

    def get_raw_text_frequencies(self, list_of_raw_text, dict_freqs=None):
        if dict_freqs is None:
            dict_freqs = self.frequencies
        data = []
        for text in list_of_raw_text:
            processed_text = self.preprocessor.process_text(text)
            pos = neg = 0
            for word in processed_text:
                pos += dict_freqs.get((word, 1),0)
                neg += dict_freqs.get((word, 0),0)
            data.append([pos, neg])
        return np.array(data)

    def get_dict_logprobabilities(self, dict_freqs=None, smoothing=True):
        if dict_freqs is None:
            dict_freqs = self.frequencies
        
        dict_word_pos_neg = self.get_word_frequencies(self.all_words)

        pos_neg = np.array(list(dict_word_pos_neg.values()))
        pos_neg_totalcount = pos_neg.sum(axis=0)

        if smoothing:
            pos_neg+=1
            pos_neg_totalcount+=len(pos_neg)

        probability_array = pos_neg/pos_neg_totalcount

        #log
        probability_array = np.log(probability_array)

        dict_positive={}
        dict_negative={}
        for word,array in zip(self.all_words,probability_array):
            dict_positive[word] = array[0]
            dict_negative[word] = array[1]

        self.dict_positive = dict_positive
        self.dict_negative = dict_negative
        return dict_positive,dict_negative

    def get_word_logprobability(self, list_of_words):
        if self.dict_positive is None:
            self.get_dict_logprobabilities()
        output = {}
        for word in list_of_words:
            output[word] = np.array([self.dict_positive[word],self.dict_negative[word]])
        return output

    def get_text_logprobabilities(self, list_of_raw_text):
        if self.dict_positive is None:
            self.get_dict_logprobabilities()
        data = []
        for text in list_of_raw_text:
            processed_text = self.preprocessor.process_text(text)
            pos = neg = 0
            for word in processed_text:
                # Adding as it is logprob:
                pos += self.dict_positive.get(word)
                neg += self.dict_negative.get(word)
            data.append([pos, neg])
        return np.array(data)