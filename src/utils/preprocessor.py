import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

class TextPreprocessor():

    def __init__(self, language='english'):
        self.language = language
        self.stemmer = SnowballStemmer(language)

    @staticmethod
    def replace_regex(text, regex, replace=''):
        return re.sub(regex, replace, text)

    @classmethod
    def remove_stock_tickers(cls, text):
        return cls.replace_regex(text,r'\$\w*')

    @classmethod
    def remove_RT(cls, text):
        return cls.replace_regex(text,r'^RT[\s]+')
    
    @classmethod
    def remove_hyperlinks(cls, text):
        return cls.replace_regex(text,r'https?:\/\/.*[\r\n]*')

    @classmethod
    def remove_hashtags(cls, text):
        return cls.replace_regex(text,r'#')

    def stem_word(self, word):
        return self.stemmer.stem(word) 

    def process_text(self, text: str, language: str = 'english'):
        """Process tweet function.

        Parameters
        ----------
        tweet: str
            text of tweet
        language: str, optional
            choose language, default is english

        Results
        -------
        processed_tweet: list
            a list of words containing the processed tweet

        """
        stopwords_language = stopwords.words(language)

        text = self.remove_stock_tickers(text)
        # remove old style retweet text "RT"
        text = self.remove_RT(text)
        # remove hyperlinks
        text = self.remove_hyperlinks(text)
        # remove hashtags
        text = self.remove_hashtags(text)
        # tokenize
        tokenizer = TweetTokenizer(preserve_case=False,
                                strip_handles=True,
                                reduce_len=True)
        text_tokens = tokenizer.tokenize(text)

        processed_text = []

        # Remove stopwords, punctuation and stem:
        for word in text_tokens:
            if (word not in stopwords_language and
                word not in string.punctuation): 
                stem_word = self.stem_word(word)  
                processed_text.append(stem_word)

        return processed_text

    def ngrams_tokenizer(self, n: int, text: str, language: str = 'english'):
        processed_text = self.process_text(text=text,language=language)
        start_token = ['<s>']*n
        end_token = ['</s>']
        return start_token + processed_text + end_token