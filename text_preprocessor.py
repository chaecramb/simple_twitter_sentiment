import json
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()
 
def tokenize(tweet):
    return tokenizer.tokenize(tweet)

def preprocess(tweet, lowercase=False):
    tokens = tokenize(tweet)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens
 
# with open('data/stream_trump.json', 'r') as f:
#     for line in f:
#         tweet = json.loads(line)
#         tokens = preprocess(tweet.get('text', ''))
    