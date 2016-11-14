from text_preprocessor import preprocess
import json
from collections import Counter
 
fpath = 'data/stream_trump.json'
with open(fpath, 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        # Create a list with all the terms
        terms_all = [term for term in preprocess(tweet.get('text',''))]
        # Update the counter
        count_all.update(terms_all)
    # Print the first 5 most frequent words
    print(count_all.most_common(10))