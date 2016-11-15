import math
from feature_extractor import terms_single, terms_cooccurences, process_tweets
from collections import defaultdict
import operator


vocab = {
    'positive_vocab': [
        'good', 'nice', 'great', 'awesome', 'outstanding',
        'fantastic', 'terrific', ':)', ':-)', 'like', 'love',
        'cool', 'ace', 'win', 'amazing', 'excited'
    ],
    'negative_vocab': [
        'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(',
        'shit', 'suck', 'rubbish', 'idiot', 'twat', 'douche',
        'horrible', 'nasty', 'vile', 'pig', 'moron', 'moronic'
    ]
}

def compute_term_probabilites(frequency_single, frequency_cooccur, number_of_tweets):
    # The probability of oberserving an individual term
    prob_term = {}
    # The probability of observing two terms occurring together
    prob_cooccur = defaultdict(lambda : defaultdict(int))
    
    for term1, occurrances in frequency_single.items():
        prob_term[term1] = occurrances / number_of_tweets
        for term2 in frequency_cooccur[term1]:
            prob_cooccur[term1][term2] = frequency_cooccur[term1][term2] / number_of_tweets

    return (prob_term, prob_cooccur)


def compute_pmi(frequency_cooccur, prob_cooccur, prob_term):
    pmi = defaultdict(lambda : defaultdict(int))
    for term1 in prob_term:
        for term2 in frequency_cooccur[term1]:
            denom = prob_term[term1] * prob_term[term2]
            pmi[term1][term2] = math.log2(prob_cooccur[term1][term2] / denom)
    return pmi

def compute_semantic_orientation(prob_term, pmi, vocab):     
    semantic_orientation = {}
    for term, _ in prob_term.items():
        positive_assoc = sum(pmi[term][positive_word] for positive_word in vocab['positive_vocab'])
        negative_assoc = sum(pmi[term][negative_word] for negative_word in vocab['negative_vocab'])
        semantic_orientation[term] = positive_assoc - negative_assoc

    return sorted(semantic_orientation.items(), 
                   key=operator.itemgetter(1), 
                   reverse=True)

frequency_single, number_of_tweets = process_tweets(terms_single, 'data/stream_trump.json')
frequency_cooccur, _ = process_tweets(terms_cooccurences, 'data/stream_trump.json') 
prob_term, prob_cooccur = compute_term_probabilites(frequency_single, 
                                                    frequency_cooccur, 
                                                    number_of_tweets)
pmi = compute_pmi(frequency_cooccur, prob_cooccur, prob_term)
semantic_sorted = compute_semantic_orientation(prob_term, pmi, vocab)

top_pos = semantic_sorted[:10]
top_neg = semantic_sorted[-10:]
 
[print(w) for w in top_pos]
[print(w) for w in top_neg]