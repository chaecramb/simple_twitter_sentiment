""" Text Pre-processor """

import json
from nltk.tokenize import TweetTokenizer
 
tokenizer = TweetTokenizer()
 
 
def tokenize(tweet):
    """ Tokenises tweet using NLTK tonenizer """
    return tokenizer.tokenize(tweet)


def preprocess(tweet, lowercase=False):
    """ Calls tokenizer and optional normalises the tweet to lowercase """
    tokens = tokenize(tweet)
    if lowercase:
        tokens = [token.lower() for token in tokens if not token.startswith('http')]
    return tokens
 
