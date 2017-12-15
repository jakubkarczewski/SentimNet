# SentimentNet (work in progress)

Deep neural network for sentiment analysis on Twitter data based on [this paper](https://arxiv.org/pdf/1508.06615.pdf) and [this tutorial](https://charlesashby.github.io/2017/06/05/sentiment-analysis-with-char-lstm/).

## Results

Network achieves a little more than 80% of validation accuracy on binary classification (positive or negative sentiment) of tweets and is capable of correctly classifying sentences such as `I like used cars` and `I used to like cars`, so it is capable of decision based on the word order.

**Work in progress**

We are currently exploring correlation between company's stock price and sentiment extracted from tweets related to this company, as this neural network might be useful in predicting stock prices. 

## Dataset

Download datasetes from [here](http://help.sentiment140.com/for-students/).
