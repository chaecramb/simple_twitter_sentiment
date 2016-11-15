import json
from nltk.tokenize import TweetTokenizer
 
tokenizer = TweetTokenizer()
 
def tokenize(tweet):
    return tokenizer.tokenize(tweet)

def preprocess(tweet, lowercase=False):
    tokens = tokenize(tweet)
    if lowercase:
        tokens = [token.lower() for token in tokens if not token.startswith('http')]
    return tokens
 
