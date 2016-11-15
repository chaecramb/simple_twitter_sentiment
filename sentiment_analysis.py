import math
from feature_extractor import terms_single, terms_cooccurences, process_tweets, terms_bigrams
from collections import defaultdict
import operator
import opinion_lexicon

vocab = opinion_lexicon.vocab

def compute_term_probabilites(frequency_terms, frequency_cooccur, number_of_tweets):
    # The probability of oberserving an individual term
    prob_term = {}
    # The probability of observing two terms occurring together
    prob_cooccur = defaultdict(lambda : defaultdict(int))
    
    for term1, occurrances in frequency_terms.items():
        prob_term[term1] = occurrances / number_of_tweets
        for term2 in frequency_cooccur[term1]:
            prob_cooccur[term1][term2] = frequency_cooccur[term1][term2] / number_of_tweets

    return (prob_term, prob_cooccur)


def compute_pmi(frequency_cooccur, prob_cooccur, prob_single, prob_bigram={}):
    pmi = defaultdict(lambda : defaultdict(int))
    for term1 in prob_bigram or prob_single:
        for term2 in frequency_cooccur[term1]:
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
