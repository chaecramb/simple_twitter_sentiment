"""Feature Extractor

This module provides functionality for extracting term frequencies from a
stream of Tweets.  

Filters are provided in order to filter 
"""

from text_preprocessor import preprocess
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
import string 
from collections import defaultdict
import operator
import opinion_lexicon


"""File path of tweet stream downloaded using twitter_stream_downloader.py""" 
fpath = 'data/stream_trump.json'


"""Build a list of stop words.

This list is used when filtering tweet in order to ignore common words and
punctuation that aren't of interest in subquent analysis.
"""
punctuation = list(string.punctuation)
ignore = ['rt', 'via', 'RT', '’', '_', '...', '..', '”', '¿', '…']
stop = stopwords.words('english') + punctuation + ignore



"""Tweet filters

A set of functions that filter individual tweets during feature exraction.
"""
def terms_single(tweet, lower=True):
    """ Return a set of the terms in a tweet i.e. each term will be included 
        only once.

    Excludes stop words.
    
    Args:
        tweet (dict): A dictionary contain tweet data from the Twitter api

    Keyword args:
        lower (bool): Defaults to true in order to normalise all terms by 
            converting to lower case.
    """
    return set(terms_all(tweet, lower=True))


def terms_hash(tweet, lower=True):
    """ Return a list of the hashtags in a tweet.
 
    Excludes stop words.   
 
    Args:
        tweet (dict): A dictionary contain tweet data from the Twitter api

    Keyword args:
        lower (bool): Defaults to true in order to normalise all terms by 
            converting to lower case.
    """

    return [term for term in preprocess(tweet.get('text',''), lowercase=lower)
              if term.startswith('#')]


def terms_only(tweet, lower=True):
    """ Return only the terms in a tweet i.e. no hashtags, or "@" mentions.
    
    Excludes stop words.

    Args:
        tweet (dict): A dictionary contain tweet data from the Twitter api

    Keyword args:
        lower (bool): Defaults to true in order to normalise all terms by 
            converting to lower case.
    """
    tokenised_tweet = preprocess(tweet.get('text',''), lowercase=lower)
    return [term for term in tokenised_tweet
            if term not in stop 
               and not term.startswith(('#', '@'))] 


def terms_all(tweet, lower=True):
    """ Return a lost of all of the terms in a tweet.

    Excluses stop words.
    
    Args:
        tweet (dict): A dictionary contain tweet data from the Twitter api

    Keyword args:
        lower (bool): Defaults to true in order to normalise all terms by 
            converting to lower case.
    """
    tokenised_tweet = preprocess(tweet.get('text',''), lowercase=lower)
    return [term for term in tokenised_tweet
            if term not in stop]


def terms_bigrams(tweet):
    """ Return all of the bigrams in a tweet.

    Excludes stop words.
    Normalises all terms by converting to lowercase. 
    
    Args:
        tweet (dict): A dictionary contain tweet data from the Twitter api 
    """
    return list(bigrams(terms_all(tweet, lower=True)))



""" Methods for processing tweet data """


def extract_frequencies(filtering_method, filepath, bigram=False):
    """ Process file containing a stream of tweets to extract term 
    frequencies.

    Args:
        fitlering_method (function): filtering method used in order to 
            select the terms that will be returned
        filepath (string): filepath for json file contain downloaded tweet data

    Keyword args:
        bigram (bool): optional flag to extract bigrams rather than single
            terms. Default to False.
    """
    term_frequency = Counter()

    number_of_tweets = 0
    with open(fpath, 'r') as f:
        for line in f: 
            number_of_tweets += 1
            tweet = json.loads(line)
            terms = filtering_method(tweet)
            term_frequency.update(terms) 

    return (term_frequency, number_of_tweets)


def __extend_cooccurrences(matrix, terms_list, bigrams_list=[]):
    """ Helper method for building the co-occurrence matrix

    Extends the matrix passed in by added the terms/bigrams to it. 

    Args:
        matrix (default): 2d defaultdict containing ints
        terms_list (list): a list of strings contain terms to be added to 
            the matrix

    Keywoard args:
        bigrams_list (list): a list of tuples representing bigrams to be 
            added to the matrix
    """
    if bigrams_list:
        for i in range(len(bigrams_list)-1):            
            for j in range(i+1, len(terms_list)):
                bigram, term = [bigrams_list[i], terms_list[j]]             
                matrix[bigram][term] += 1
    else:
        for i in range(len(terms_list)-1):            
            for j in range(i+1, len(terms_list)):
                term1, term2 = [terms_list[i], terms_list[j]]             
                matrix[term1][term2] += 1        
    return matrix


def build_cooccurrences(filepath, bigram=False):
    """ Building the co-occurrence matrix based upon tweet data
 
    Builds the matrix such that matrix[term1][term1] contains the number of
    times term1 and term2 have occured together.

    If an optional bigrams list is passed in the for will be: 
        matrix[bigram][term]

    Args:
        filepath (string): filepath json file contain downloaded tweet data

    Keywoard args:
        bigram (bool): creates matrix with bigrams for the first dimension
            if True. False by default
    """
    term_frequency = defaultdict(lambda : defaultdict(int))
    # TODO: look into scipy.sparse to use sparse matrix instead of defaultdict


    with open(fpath, 'r') as f:
        for line in f: 
            tweet = json.loads(line)
            terms = terms_only(tweet)
            bigrams_list = []
            if bigram: 
                bigrams_list = terms_bigrams(tweet)
            term_frequency = __extend_cooccurrences(term_frequency, 
                                                    terms, 
                                                    bigrams_list)
    return term_frequency

