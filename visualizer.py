from feature_extractor import process_tweets, terms_only
import vincent

def prepare_bar_chart_data(filter_method, filepath):
    """ Process tweets for visualation in a bar chart using D3.
    Uses Vincent to translate the data in Vega (usable by D3).

    Args:
        filter_method (function): filtering method to be used during feature
            extraction
        filepath (string): path to Tweet data from the Twitter API
    """
    word_freq = process_tweets(filter_method, filepath)[0].most_common(20)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('web/term_freq.json')

prepare_bar_chart_data(terms_only, 'data/stream_trump.json')