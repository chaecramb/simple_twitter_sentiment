from feature_extractor import *
from sentiment_analysis import *
import operator


if __name__ == '__main__':
    print('*********************************************')
    print('Calculating word frequencies')
    print('*********************************************')
    print()
    print('Extracting features from tweets...')
    print()    

    common_bigrams = process_tweets(terms_bigrams, 'data/stream_trump.json')[0].most_common(20)
    common_terms = process_tweets(terms_only, 'data/stream_trump.json')[0].most_common(20)
  
    print('Most common bigrams')
    [print(str(i+1) + ':', a, b + ':', c) for i, ((a,b), c) in enumerate(common_bigrams)]
    print()

    print('Most common single terms')
    [print(str(i+1) + ':', t + ':', c) for i, (t, c) in enumerate(common_terms)]
    print()

    print('*********************************************')
    print('Calculating sentiment')
    print('*********************************************')
    print()
    
    print('Calculating frequency of single terms...')
    frequency_single, number_of_tweets = process_tweets(terms_single, 'data/stream_trump.json')
    print('Calculating frequency of bigrams...')
    frequency_bigrams, number_of_bigrams = process_tweets(terms_bigrams, 'data/stream_trump.json', bigram=True)
    print('Calculating frequency of co-occurrence matrix for single terms...')
    frequency_cooccur_single, _ = process_tweets(terms_cooccurences, 'data/stream_trump.json') 
    print('Calculating frequency of co-occurrence matrix for bigrams...')
    frequency_cooccur_bigrams, _ = process_tweets(terms_cooccurences, 'data/stream_trump.json', bigram=True) 

    print('Calculating probabilities for single terms...')
    prob_single, prob_cooccur_single = compute_term_probabilites(frequency_single, 
                                                          frequency_cooccur_single, 
                                                          number_of_tweets)
    print('Calculating probabilities for bigrams...')
    prob_bigrams, prob_cooccur_bigrams = compute_term_probabilites(frequency_bigrams, 
                                              frequency_cooccur_bigrams, 
                                              number_of_tweets)
 
    print('Calculating pointwise mutual information for single terms...')
    single_pmi = compute_pmi(frequency_cooccur_single, prob_cooccur_single, prob_single)
    print('Calculating pointwise mutual information for bigrams...')
    bigrams_pmi = compute_pmi(frequency_cooccur_bigrams, prob_cooccur_bigrams, prob_single, prob_bigrams)

    print('Calculating semantic orientation for single terms...')
    semantic_orientation_single = compute_semantic_orientation(prob_single, single_pmi, vocab)
    print('Calculating semantic orientation for bigrams...')
    semantic_orientation_bigrams = compute_semantic_orientation(prob_bigrams, bigrams_pmi, vocab)
    
    print()
    print('Results:')
    print()
    print('Opinion on specifc terms:')   
    print('Barack Obama:', semantic_orientation_bigrams.get(('barack', 'obama'), 'not present'))
    print('Donald Trump:', semantic_orientation_bigrams.get(('donald', 'trump'), 'not present'))
    print('York Times:', semantic_orientation_bigrams.get(('york', 'times'), 'not present'))
    print('Melania Trump:', semantic_orientation_bigrams.get(('melania', 'trump'), 'not present'))
    print('Steve Bannon:', semantic_orientation_bigrams.get(('steve', 'bannon'), 'not present'))
    print('Trump sticking:', semantic_orientation_bigrams.get(('trump', 'sticking'), 'not present'))
    print('Trump win:', semantic_orientation_bigrams.get(('trump', 'win'), 'not present'))
    print("politics Trump's:", semantic_orientation_bigrams.get(('politics', "trump's"), 'not present'))
    print("Trump's cabinet:", semantic_orientation_bigrams.get(("trump's", 'cabinet'), 'not present'))
    print('Trump:', semantic_orientation_single.get('trump', 'not present'))
    print('Donald:', semantic_orientation_single.get('donald', 'not present'))
    print('Obama:', semantic_orientation_single.get('obama', 'not present'))
    print('Pence:', semantic_orientation_single.get('pence', 'not present'))
    print('Michelle:', semantic_orientation_single.get('michelle', 'not present'))
    print('Melania:', semantic_orientation_single.get('melania', 'not present'))

    semantic_sorted_single = sorted(semantic_orientation_single.items(), 
                       key=operator.itemgetter(1), 
                       reverse=True)
    semantic_sorted_bigrams = sorted(semantic_orientation_bigrams.items(), 
                       key=operator.itemgetter(1), 
                       reverse=True)

    top_pos_single = [b for b in semantic_sorted_single
                        if b[0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0] not in opinion_lexicon.vocab['negative_vocab']
                     ][:20] 
    top_neg_single = [b for b in semantic_sorted_single
                        if b[0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0] not in opinion_lexicon.vocab['negative_vocab']
                      ][:-21:-1] 

    top_pos_bigrams = [b for b in semantic_sorted_bigrams 
                        if b[0][0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][0] not in opinion_lexicon.vocab['negative_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['negative_vocab']
                      ][:20]
    top_neg_bigrams = [b for b in semantic_sorted_bigrams
                        if b[0][0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][0] not in opinion_lexicon.vocab['negative_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['negative_vocab']
                      ][:-21:-1]

    print()
    print('Most positive bigrams:') 
    [print(str(i+1) + ':', a, b + ':', p) for i, ((a,b),p) in enumerate(top_pos_bigrams)]
    print()
    print('Most negative bigrams:') 
    [print(str(i+1) + ':', a, b + ':', p) for i, ((a,b),p) in enumerate(top_neg_bigrams)]
    print()
    print('Most positive single terms:') 
    [print(str(i+1) + ':', t + ':', p) for i, (t,p) in enumerate(top_pos_single)]
    print()
    print('Most negative single terms:') 
    [print(str(i+1) + ':', t + ':', p) for i, (t,p) in enumerate(top_neg_single)]
