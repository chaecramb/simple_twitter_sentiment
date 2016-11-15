from feature_extractor import process_tweets, terms_only
import vincent

word_freq = process_tweets(terms_only, 'data/stream_trump.json')[0].most_common(20)
labels, freq = zip(*word_freq)
data = {'data': freq, 'x': labels}
bar = vincent.Bar(data, iter_idx='x')
bar.to_json('term_freq.json')