# Twitter sentiment analysis tools

A simple sentiment analysis app written as a learning exercise. Credit to Marco Bonzanini for his [tutorial](https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/) which I worked through, adapted and extended based upon my own interests. 

There's a significant amount more that could/should be done to it, but I will probably leave it here as it has served its purpose of allowing me to experiment with the methods involved. I plan to follow up with a more sophisticated version.

The semantic orientation of a phrase is calculated as the mutual information between the given phrase and "positive words" minus the mutual information between the given phrase and "negative words". It is an adaptation of the unsupervised learning algorithm in Peter D. Turney's paper [Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews](https://arxiv.org/abs/cs/0212032).

Positive and negative words are taken from the [opinion lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon) of 6800 words provided by Hu and Liu, KDD-2004.


## Usage

### Downloading tweets

In order to download Tweets run `python twitter_stream_download.py -d [directory] -q [query]` (Twitter app registration is required in order to provide the credentials - currently set up to be read from the environment)

Tweets containing the query term will be saved to a json file in the specified directory until the script is terminated. 

### Sentiment analysis

Run `sentiment_analysis.py` to perform analysis of the downloaded tweets (filepath is currently hard-coded, so will need to modified), and save to a log file within the log folder. Specific terms of interest will be read from `terms_of_interest.py` and added to the analysis. Semantic orientation with be record for these terms along with the top 20 most positive and negative terms (both for single words and for bigrams).


