from feature_extractor import process_tweets, terms_only
import vincent

def prepare_bar_chart_data(filter_method, filepath):
    word_freq = process_tweets(filter_method, filepath)[0].most_common(20)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('web/term_freq.json')

prepare_bar_chart_data(terms_only, 'data/stream_trump.json')