# Sentiment Analysis Using More than 150K Tweets

## 1 - Introduction

Sentiment analysis involves assessing opinions through natural language processing, text analysis, and statistical methods. Many companies now monitor customer emotions—what they express, how they express it, and what their words imply. Rather than manually interpreting each word to determine sentiment, advancements in machine learning have made it possible for machines to analyze news or comments and quickly identify the sentiment or meaning behind a sentence.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*fDnVCDLv3a8tyxuZEWIS3w.png" />
</p>
<p align="center">
  Source: <a href="https://medium.com/@liangnguyen612/sentiment-analysis-in-python-81-accuracy-ab5d694b7ef8">link</a>
</p>

One of the opinion-based social networks where sentiments tend to be very divisive is Twitter. Understanding the sentiment around specific topics is essential for focusing content.

However, sometimes this analysis needs to be performed automatically due to the large volume of tweets to be analyzed. One of the most commonly used and interesting methods for conducting this study is by using Natural Language Processing (NLP), a field of artificial intelligence that deals with the interaction between computers and human language.

Therefore, this study develops a Python script to perform sentiment analysis on tweets.

## 2 - Methodology

### 2.1 - Data

For this study, a dataset available on Kaggle was used, accessible through this link: [Twitter and Reddit Sentimental analysis Dataset
](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)

### 2.2 - Python Version and Libraries

To create the code, we use: 

- [Python](https://www.python.org/) 3.9.2
  - [pandas](https://pandas.pydata.org/) 1.5.3
  - [sci-kit learn](https://scikit-learn.org/stable/) 1.5.1
  - [textblob](https://textblob.readthedocs.io/en/dev/#) 0.18.0

## 3 - Implementation

Primeiramente, importamos e utilizamos a biblioteca pandas para extrair 
