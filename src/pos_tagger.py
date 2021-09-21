import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer



class POSTagger():

    def __init__(self, language='english'):
        self.language = language

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self,language):
        self._language = language

        if language == 'english':
            # Morphology rules used to assign unknown word tokens
            self.noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
            self.verb_suffix = ["ate", "ify", "ise", "ize"]
            self.adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
            self.adv_suffix = ["ward", "wards", "wise"]
        elif language == 'spanish':
            self.noun_suffix = ["ion",'dad']
            self.verb_suffix = ["ar", "er", "ir","or"]
            self.adj_suffix = ['able','ible','aceo','acea','ado','ido','ador','al','ano','ana','ante','ente','ar',
                                'ario','aria','atico','atica','bio','bundo','bunda','crata','dizo','diza','dor',
                                'ento','ero','estre','fero','fera','fico','filo','fila','fugo','fuga','icio','ico',
                                'iego','iega','iento', 'ienta','il','ino','ista','istico','ivo','izo','morfo','on','ona',
                                'oso','osa','torio','toria','udo','uno','una','voro','vora']
            self.adv_suffix = ["mente"]
        else:
            self.noun_suffix = [] 
            self.verb_suffix = []
            self.adj_suffix = []
            self.adv_suffix = []

    def assign_unk(self, tok):
        """
        Assign unknown word tokens
        """
        # Digits
        if any(char.isdigit() for char in tok):
            return "--unk_digit--"

        # Punctuation
        elif any(char in string.punctuation for char in tok):
            return "--unk_punct--"

        # Upper-case
        elif any(char.isupper() for char in tok):
            return "--unk_upper--"

        # Nouns
        elif any(tok.endswith(suffix) for suffix in self.noun_suffix):
            return "--unk_noun--"

        # Verbs
        elif any(tok.endswith(suffix) for suffix in self.verb_suffix):
            return "--unk_verb--"

        # Adjectives
        elif any(tok.endswith(suffix) for suffix in self.adj_suffix):
            return "--unk_adj--"

        # Adverbs
        elif any(tok.endswith(suffix) for suffix in self.adv_suffix):
            return "--unk_adv--"

        return "--unk--"

    def get_word_tag(self, word_tag, vocabulary): 
        word, tag = word_tag
        if word not in vocabulary: 
            # Handle unknown words
            word = self.assign_unk(word)
            return word, tag
        else:
            return word, tag