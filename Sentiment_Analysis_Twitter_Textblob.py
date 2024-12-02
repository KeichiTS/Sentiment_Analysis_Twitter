# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:37:22 2024

@author: KeichiTS
"""

import pandas as pd 
from sklearn.metrics import accuracy_score
from textblob import TextBlob


def analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity

def score_analysis(score):
  if score >= .2:
    return 1
  elif score < .2 and score > -.2:
    return 0
  return -1

df = pd.read_csv('Twitter_Data.csv')
df = df.dropna()
df.head()

df['new_review_textblop'] = df['clean_text'].apply(analysis)

df['new_sentiment_textblop'] = df['new_review_textblop'].apply(score_analysis)

accuracy_textblop = accuracy_score(df['category'], df['new_sentiment_textblop'])
print('The accuracy of this model was:', accuracy_textblop)