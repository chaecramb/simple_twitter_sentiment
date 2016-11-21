""" Sentiment logger

Runs feature extraction and sentiment analysis on a given file and logs
the results. 
"""

from feature_extractor import *
from sentiment_analysis import *
from datetime import datetime
import operator
import terms_of_interest

# TODO: extract hardcoded path to command line argument
filepath = 'log/sentiment_log.txt'

with open(filepath, "a") as f:
    f.write(str(datetime.now()) + '\n\n')

    print('Extracting features from tweets...')  
    common_bigrams = extract_frequencies(terms_bigrams, 
                     'data/stream_trump.json')[0].most_common(20)
    common_terms = extract_frequencies(terms_only, 
                   'data/stream_trump.json')[0].most_common(20)

    print('Calculating frequency of single terms...')
    frequency_single, number_of_tweets = extract_frequencies(terms_single, 
                                                        'data/stream_trump.json')

    print('Calculating frequency of bigrams...')
    frequency_bigrams, number_of_bigrams = extract_frequencies(terms_bigrams, 
                                                         'data/stream_trump.json', 
                                                         bigram=True)

    print('Creating frequency of co-occurrence matrix for single terms...')
    frequency_cooccur_single = build_cooccurrences('data/stream_trump.json') 

    print('Creating frequency of co-occurrence matrix for bigrams...')
    frequency_cooccur_bigrams = build_cooccurrences('data/stream_trump.json', 
                                                       bigram=True) 

    print('Calculating probabilities for single terms...')
    prob_single, prob_cooccur_single = compute_term_probabilites(frequency_single, 
                                                          frequency_cooccur_single, 
                                                          number_of_tweets)

    print('Calculating probabilities for bigrams...')
    prob_bigrams, prob_cooccur_bigrams = compute_term_probabilites(frequency_bigrams, 
                                              frequency_cooccur_bigrams, 
                                              number_of_tweets)

    print('Calculating pointwise mutual information for single terms...')
    single_pmi = compute_pmi(frequency_cooccur_single, 
                             prob_cooccur_single, 
                             prob_single)
  
    print('Calculating pointwise mutual information for bigrams...')
    bigrams_pmi = compute_pmi(frequency_cooccur_bigrams, 
                              prob_cooccur_bigrams, 
                              prob_single, 
                              prob_bigrams)

    print('Calculating semantic orientation for single terms...')
    semantic_orientation_single = compute_semantic_orientation(prob_single, 
                                                               single_pmi, 
                                                               vocab)
 
    print('Calculating semantic orientation for bigrams...')
    semantic_orientation_bigrams = compute_semantic_orientation(prob_bigrams, 
                                                                bigrams_pmi, 
                                                                vocab)
  
    print('Sorting semantic orientation for single terms...')
    semantic_sorted_single = sorted(semantic_orientation_single.items(), 
                       key=operator.itemgetter(1), 
                       reverse=True)
 
    print('Sorting semantic orientation for bigrams...')
    semantic_sorted_bigrams = sorted(semantic_orientation_bigrams.items(), 
                       key=operator.itemgetter(1), 
                       reverse=True)

    print('Extracting most positive single terms...')
    top_pos_single = [b for b in semantic_sorted_single
                        if b[0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0] not in opinion_lexicon.vocab['negative_vocab']
                     ][:20] 
    print('Extracting most negative single terms...')
    top_neg_single = [b for b in semantic_sorted_single
                        if b[0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0] not in opinion_lexicon.vocab['negative_vocab']
                      ][:-21:-1] 
    print('Extracting most positive bigrams...')
    top_pos_bigrams = [b for b in semantic_sorted_bigrams 
                        if b[0][0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][0] not in opinion_lexicon.vocab['negative_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['negative_vocab']
                      ][:20]
    
    print('Extracting most negative bigrams...')
    top_neg_bigrams = [b for b in semantic_sorted_bigrams
                        if b[0][0] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['positive_vocab'] 
                        and b[0][0] not in opinion_lexicon.vocab['negative_vocab'] 
                        and b[0][1] not in opinion_lexicon.vocab['negative_vocab']
                      ][:-21:-1]

    print('Writing to {0}...'.format(filepath))

    f.write('Number of tweets analysed: {0}\n\n'.format(number_of_tweets))

    f.write('Opinion on terms of interest\n')
    f.write('Term | Semantic Orientation\n') 
    for t1, t2 in terms_of_interest.terms['bigrams']:
        f.write("{0} {1} | {2}\n".format(t1, t2, str(semantic_orientation_bigrams.get((t1, t2), 'not present'))))

    for term in terms_of_interest.terms['single']:
        f.write("{0} | {1}\n".format(term, str(semantic_orientation_single.get((term), 'not present'))))

    f.write('\n')

    f.write('Most positive single terms\n') 
    f.write('Position | Term | Semantic Orientation\n') 
    [f.write("{0} | {1} | {2}\n".format(i+1, t, p)) 
     for i, (t, p) 
     in enumerate(top_pos_single)]
    f.write('\n')

    f.write('Most negative single terms\n') 
    f.write('Position | Term | Semantic Orientation\n') 
    [f.write("{0} | {1} | {2}\n".format(i+1, t, p)) 
     for i, (t, p) 
     in enumerate(top_neg_single)]
    f.write('\n')

    f.write('Most positive bigrams\n')
    f.write('Position | Terms | Semantic Orientation\n') 
    [f.write("{0} | {1} {2} | {3}\n".format(i+1, t1, t2, p)) 
     for i, ((t1, t2), p) 
     in enumerate(top_pos_bigrams)]
    f.write('\n')
   
    f.write('Most negative bigrams\n') 
    f.write('Position | Terms | Semantic Orientation\n') 
    [f.write("{0} | {1} {2} | {3}\n".format(i+1, t1, t2, p)) 
     for i, ((t1, t2), p) 
     in enumerate(top_neg_bigrams)]
    f.write('\n')
   
    f.write('Most common single terms\n')
    f.write('Position | Term | Count | Semantic Orientation\n') 
    [f.write('{0} | {1} | {2} | {3}\n'.format(i+1, t, c, semantic_orientation_single.get(t, 'not present'))) 
     for i, (t, c) 
     in enumerate(common_terms)]
    f.write('\n')
   
    f.write('Most common bigrams\n')
    f.write('Position | Terms | Count | Semantic Orientation\n') 
    [f.write('{0} | {1} {2} | {3} | {4}\n'.format(i+1,  t1, t2, c, semantic_orientation_bigrams.get((t1, t2), 'not present'))) 
     for i, ((t1, t2), c) 
     in enumerate(common_bigrams)]
    f.write('\n**********\n\n')
