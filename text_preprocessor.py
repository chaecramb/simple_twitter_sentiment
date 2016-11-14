import json
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
 
punctuation = list(string.punctuation)
ignore = ['rt', 'via', 'RT', '’', '_', '...', '..', '”', '¿', '…']
stop = stopwords.words('english') + punctuation + ignore
tokenizer = TweetTokenizer()
 
def tokenize(tweet):
    return tokenizer.tokenize(tweet)

def preprocess(tweet, lowercase=False):
    tokens = tokenize(tweet)
    tokens = [token for token in tokens if token not in stop]
    if lowercase:
        tokens = [token.lower() for token in tokens if token not in stop]
    return tokens
 
