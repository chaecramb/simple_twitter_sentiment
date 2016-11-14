import json
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()
 
with open('data/stream_trump.json', 'r') as f:
    line = f.readline() # read only the first tweet/line
    tweet = json.loads(line) # load it as Python dict
    print(tknzr.tokenize(tweet['text']))