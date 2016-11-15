from feature_extractor import *
from sentiment_analysis import *
from datetime import datetime
import operator

with open("sentiment_log.txt", "a") as f:
    f.write(str(datetime.now()) + '\n\n')

    print('Extracting features from tweets...')  
    common_bigrams = process_tweets(terms_bigrams, 'data/stream_trump.json')[0].most_common(20)
    common_terms = process_tweets(terms_only, 'data/stream_trump.json')[0].most_common(20)

    f.write('Most common bigrams\n')
    [f.write('{0}: {1} {2}: {3}\n'.format(i+1, a, b, c)) for i, ((a,b), c) in enumerate(common_bigrams)]
    f.write('\n')

    f.write('Most common single terms\n')
    [f.write('{0}: {1}: {2}\n'.format(i+1, t, c)) for i, (t, c) in enumerate(common_terms)]
    f.write('\n')

    print('Calculating frequency of single terms...')
    frequency_single, number_of_tweets = process_tweets(terms_single, 'data/stream_trump.json')
    print('Calculating frequency of bigrams...')
    frequency_bigrams, number_of_bigrams = process_tweets(terms_bigrams, 'data/stream_trump.json', bigram=True)
    print('Creating frequency of co-occurrence matrix for single terms...')
    frequency_cooccur_single, _ = process_tweets(terms_cooccurences, 'data/stream_trump.json') 
    print('Creating frequency of co-occurrence matrix for bigrams...')
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

    f.write('Number of tweets analysed: {0}\n\n'.format(number_of_tweets))
    f.write('Opinion on specifc terms:\n')   
    f.write('Barack Obama: ' + str(semantic_orientation_bigrams.get(('barack', 'obama'), 'not present')) + '\n')
    f.write('Donald Trump: ' + str(semantic_orientation_bigrams.get(('donald', 'trump'), 'not present')) + '\n')
    f.write('York Times: ' + str(semantic_orientation_bigrams.get(('york', 'times'), 'not present')) + '\n')
    f.write('Melania Trump: ' + str(semantic_orientation_bigrams.get(('melania', 'trump'), 'not present')) + '\n')
    f.write('Steve Bannon: ' + str(semantic_orientation_bigrams.get(('steve', 'bannon'), 'not present')) + '\n')
    f.write('Trump sticking: ' + str(semantic_orientation_bigrams.get(('trump', 'sticking'), 'not present')) + '\n')
    f.write('Trump win: ' + str(semantic_orientation_bigrams.get(('trump', 'win'), 'not present')) + '\n')
    f.write("politics Trump's: " + str(semantic_orientation_bigrams.get(('politics', "trump's"), 'not present')) + '\n')
    f.write("Trump's cabinet: " + str(semantic_orientation_bigrams.get(("trump's", 'cabinet'), 'not present')) + '\n')
    f.write('Trump: ' + str(semantic_orientation_single.get('trump', 'not present')) + '\n')
    f.write('Donald: ' + str(semantic_orientation_single.get('donald', 'not present')) + '\n')
    f.write('Obama: ' + str(semantic_orientation_single.get('obama', 'not present')) + '\n')
    f.write('Pence: ' + str(semantic_orientation_single.get('pence', 'not present')) + '\n')
    f.write('Michelle: ' + str(semantic_orientation_single.get('michelle', 'not present')) + '\n')
    f.write('Melania: ' + str(semantic_orientation_single.get('melania', 'not present')) + '\n')

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

    f.write('\n')
    f.write('Most positive bigrams:\n') 
    [f.write("{0}: {1} {2}: {3}\n".format(i+1,a,b,p)) for i, ((a,b),p) in enumerate(top_pos_bigrams)]
    f.write('\n')
    f.write('Most negative bigrams:\n') 
    [f.write("{0}: {1} {2}: {3}\n".format(i+1,a,b,p)) for i, ((a,b),p) in enumerate(top_neg_bigrams)]
    f.write('\n')
    f.write('Most positive single terms:\n') 
    [f.write("{0}: {1}: {2}\n".format(i+1,t,p)) for i, (t,p) in enumerate(top_pos_single)]
    f.write('\n')
    f.write('Most negative single terms:\n') 
    [f.write("{0}: {1}: {2}\n".format(i+1,t,p)) for i, (t,p) in enumerate(top_neg_single)]
    f.write('\n**********\n')
