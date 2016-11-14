from text_preprocessor import preprocess
import json
from collections import Counter
from nltk.corpus import stopwords
import string 

fpath = 'data/stream_trump.json'

punctuation = list(string.punctuation)
ignore = ['rt', 'via', 'RT', '’', '_', '...', '..', '”', '¿', '…']
stop = stopwords.words('english') + punctuation + ignore

# Count terms only once, equivalent to Document Frequency
def terms_single(tweet):
    return set(terms_all)

# Count hashtags only
def terms_hash(tweet):
    return [term for term in preprocess(tweet.get('text',''))
              if term.startswith('#')]

# Count terms only (no hashtags, no mentions)
def terms_only(tweet):
    return [term for term in preprocess(tweet.get('text','')) 
              if term not in stop and
              not term.startswith(('#', '@'))] 

def terms_all(tweet):
    return [term for term in preprocess(tweet.get('text',''))
              if term not in stop]

with open(fpath, 'r') as f:
    count = 0
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        # Create a list with all the terms
        terms = terms_all(tweet)
        # Update the counter
        count_all.update(terms)
        count += 1
    # Print the first 5 most frequent words
    print(count)
    print(count_all.most_common(50))
