# SentimentNet (work in progress)

Deep neural network for sentiment analysis on Twitter data based on [this paper](https://arxiv.org/pdf/1508.06615.pdf) and [this tutorial](https://charlesashby.github.io/2017/06/05/sentiment-analysis-with-char-lstm/).

## Results

Network achieves a little more than 80% of validation accuracy on binary classification (positive or negative sentiment) of tweets and is capable of correctly classifying sentences such as `I like used cars` and `I used to like cars`, so it is capable of decision based on the word order.

**Work in progress**

![alt text](https://github.com/jakubkarczewski/SentimNet/blob/master/charts/google/cena_i_sent_dla_Google.png "100x100")

We are currently exploring correlation between company's stock price and sentiment extracted from tweets related to this company, as this neural network might be useful in predicting stock prices. So far, content of this repository has been used as a part of engineering thesis project by one of the authors. Thesis can be found [here](https://drive.google.com/file/d/1qKr3eurZMnV9fHgVFisPL_mB73luzb86/view?usp=sharing).

## Dataset

Download datasetes from [here](http://help.sentiment140.com/for-students/).
