from text_preprocessor import preprocess
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams 
import string 
from collections import defaultdict
import operator

fpath = 'data/stream_trump.json'

punctuation = list(string.punctuation)
ignore = ['rt', 'via', 'RT', '’', '_', '...', '..', '”', '¿', '…']
stop = stopwords.words('english') + punctuation + ignore

# Count terms only once, equivalent to Document Frequency
def terms_single(tweet, lower=True):
    return set(terms_all(tweet, lower=True))

# Count hashtags only
def terms_hash(tweet, lower=True):
    return [term for term in preprocess(tweet.get('text',''), lowercase=lower)
              if term.startswith('#')]

# Count terms only (no hashtags, no mentions)
def terms_only(tweet, lower=True):
    return [term for term in preprocess(tweet.get('text',''), lowercase=lower) 
              if term not in stop and
              not term.startswith(('#', '@'))] 

def terms_all(tweet, lower=True):
    return [term for term in preprocess(tweet.get('text',''), lowercase=lower)
              if term not in stop]

def terms_bigrams(tweet):
    return bigrams(terms_all(tweet, lower=True))

def build_cooccurrence_matrix(matrix, terms):
    for i in range(len(terms)-1):            
        for j in range(i+1, len(terms)):
            w1, w2 = sorted([terms[i], terms[j]])                
            if w1 != w2:
                matrix[w1][w2] += 1
    return matrix

def terms_cooccurences(cooccur_matrix):
    most_common = []
    # For each term, look for the most common co-occurrent terms
    for term1 in cooccur_matrix:
        term1_max_terms = sorted(cooccur_matrix[term1].items(), 
                               key=operator.itemgetter(1), 
                               reverse=True)[:5]
        for term2, term2_count in term1_max_terms:
            most_common.append(((term1, term2), term2_count))
    # Return the most frequent co-occurrences
    return sorted(most_common, key=operator.itemgetter(1), reverse=True)

def process_tweets(filtering_method, filepath):
    cooccurences = filtering_method == terms_cooccurences

    if cooccurences:
        # look into scipy.sparse to use sparse matrix instead of defaultdict
        term_frequency = defaultdict(lambda : defaultdict(int))
    else:
        term_frequency = Counter()

    with open(fpath, 'r') as f:
        for line in f: 
            tweet = json.loads(line)
            terms = terms_only(tweet) if cooccurences else filtering_method(tweet)

            if cooccurences:
                term_frequency = build_cooccurrence_matrix(term_frequency, terms)
            else:
                term_frequency.update(terms) 

    return terms_cooccurences(term_frequency) if cooccurences else term_frequency 
