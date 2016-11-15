import math
import feature_extractor
from collections import defaultdict

print(feature_extractor.process_tweets(feature_extractor.terms_all, 'data/stream_trump.json'))

# # n_docs is the total n. of tweets
# p_t = {}
# p_t_com = defaultdict(lambda : defaultdict(int))
 
# for term, n in count_stop_single.items():
#     p_t[term] = n / n_docs
#     for t2 in com[term]:
#         p_t_com[term][t2] = com[term][t2] / n_docs

# positive_vocab = [
#     'good', 'nice', 'great', 'awesome', 'outstanding',
#     'fantastic', 'terrific', ':)', ':-)', 'like', 'love',
#     'cool', 'ace', 'win', 'amazing', 'excited'
#     # shall we also include game-specific terms?
#     # 'triumph', 'triumphal', 'triumphant', 'victory', etc.
# ]
# negative_vocab = [
#     'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(',
#     'shit', 'suck', 'rubbish', 'idiot', 'twat', 'douche',
#     'horrible', 'nasty', 'vile', 'pig', 'moron', 'moronic'
#     # 'defeat', etc.
# ]

# pmi = defaultdict(lambda : defaultdict(int))
# for t1 in p_t:
#     for t2 in com[t1]:
#         denom = p_t[t1] * p_t[t2]
#         pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)
 
# semantic_orientation = {}
# for term, n in p_t.items():
#     positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
#     negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
#     semantic_orientation[term] = positive_assoc - negative_assoc

#     semantic_sorted = sorted(semantic_orientation.items(), 
#                              key=operator.itemgetter(1), 
#                              reverse=True)

# top_pos = semantic_sorted[:100]
# top_neg = semantic_sorted[-100:]
 
# [print(w) for w in top_pos]
# [print(w) for w in top_pos]