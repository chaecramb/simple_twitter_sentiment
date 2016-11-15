import math
from feature_extractor import terms_single, terms_cooccurences, process_tweets, terms_bigrams
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

if __name__ == '__main__':
    frequency_single, number_of_tweets = process_tweets(terms_single, 'data/stream_trump.json')
    frequency_bigrams, number_of_bigrams = process_tweets(terms_bigrams, 'data/stream_trump.json', bigram=True)
    frequency_cooccur_single, _ = process_tweets(terms_cooccurences, 'data/stream_trump.json') 
    frequency_cooccur_bigrams, _ = process_tweets(terms_cooccurences, 'data/stream_trump.json', bigram=True) 


    prob_single, prob_cooccur_single = compute_term_probabilites(frequency_single, 
                                                          frequency_cooccur_single, 
                                                          number_of_tweets)

    prob_bigrams, prob_cooccur_bigrams = compute_term_probabilites(frequency_bigrams, 
                                              frequency_cooccur_bigrams, 
                                              number_of_tweets)
    single_pmi = compute_pmi(frequency_cooccur_single, prob_cooccur_single, prob_single)
    bigrams_pmi = compute_pmi(frequency_cooccur_bigrams, prob_cooccur_bigrams, prob_single, prob_bigrams)


    semantic_orientation_single = compute_semantic_orientation(prob_single, single_pmi, vocab)
    semantic_orientation_bigrams = compute_semantic_orientation(prob_bigrams, bigrams_pmi, vocab)
    semantic_sorted_single = sorted(semantic_orientation_single.items(), 
                       key=operator.itemgetter(1), 
                       reverse=True)
    semantic_sorted_bigrams = sorted(semantic_orientation_bigrams.items(), 
                       key=operator.itemgetter(1), 
                       reverse=True)

    top_pos_single = semantic_sorted_single[:20]
    top_neg_single = semantic_sorted_single[-20:]

    top_pos_bigrams = semantic_sorted_bigrams[:20]
    top_neg_bigrams = semantic_sorted_bigrams[-20:]
     
    print('Barack Obama approval:', semantic_orientation_bigrams.get(('barack', 'obama'), 'not present'))
    print('Donald Trump approval:', semantic_orientation_bigrams.get(('donald', 'trump'), 'not present'))
    print('White nationalism approval:', semantic_orientation_bigrams.get(('white', 'nationalism'), 'not present'))
    print()
    print('Most positive bigrams:') 
    [print(str(i+1) + ':', a, b + ':', p) for i, ((a,b),p) in enumerate(top_pos_bigrams)]
    print()
    print('Most negative bigrams:') 
    [print(str(i+1) + ':', a, b + ':', p) for i, ((a,b),p) in enumerate(top_neg_bigrams)]
    print()
    print('Most positive single:') 
    [print(str(i+1) + ':', t + ':', p) for i, (t,p) in enumerate(top_pos_single)]
    print()
    print('Most negative single:') 
    [print(str(i+1) + ':', t + ':', p) for i, (t,p) in enumerate(top_neg_single)]
