""" Tools for sentiment analysis """

import math
from feature_extractor import terms_single, terms_bigrams
from collections import defaultdict
import operator
import opinion_lexicon


""" Dict of positive and negative vocab """
vocab = opinion_lexicon.vocab


def compute_probabilites(frequencies, cooccur_matrix, number_of_tweets):
    """ Calculates the probabilty of oberservating individual terms or bigrams
    and of individual terms or bigrams co-occurring with another term.

    Args:
        freqencies (dict): Dictionary of term or bigram frequencies
        cooccur_matrix (dict): 2d dict matrix of two terms, or a bigram and a
        term, and a count of their co-occurrences

    Returns:
    tuple
        prob_term : a dict of terms and their probabilites
        prob_cooccur : a dict matrix of two terms (or a bigram and a term) 
        and their probability of co-occurring
    """

    prob_term = {}
    prob_cooccur = defaultdict(lambda : defaultdict(int))
    
    for term1, occurrances in frequencies.items():
        # Calculate the probability of a term occurring
        prob_term[term1] = occurrances / number_of_tweets
        for term2 in cooccur_matrix[term1]:
            # Calculate the probability of term1 co-occurring with term2
            prob_cooccur[term1][term2] = cooccur_matrix[term1][term2] / number_of_tweets

    return (prob_term, prob_cooccur)


def compute_pmi(cooccur_matrix, prob_cooccur, prob_single, prob_bigram={}):
    pmi = defaultdict(lambda : defaultdict(int))
    for term1 in prob_bigram or prob_single:
        for term2 in cooccur_matrix[term1]:
            if prob_bigram:
                denom = prob_bigram[term1] * prob_single[term2]
            else:
                denom = prob_single[term1] * prob_single[term2]    
            pmi[term1][term2] = math.log2(prob_cooccur[term1][term2] / denom)
    return pmi


def compute_semantic_orientation(prob_term, pmi, vocab):     
    semantic_orientation = {}
    for term, _ in prob_term.items():
        positive_assoc = sum(pmi[term][positive_word] for positive_word in vocab['positive_vocab'])
        negative_assoc = sum(pmi[term][negative_word] for negative_word in vocab['negative_vocab'])
        semantic_orientation[term] = positive_assoc - negative_assoc
    return semantic_orientation
