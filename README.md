# Sentiment Analysis Using More than 150K Tweets
By: KeichiTS
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

First, we import and use the pandas library to read the CSV directly from the local folder and store it in a well-structured dataframe. It is also necessary at this stage to handle the NA values present in the sample. In this case, it was decided to simply remove the rows with missing information.
```
import pandas as pd 

df = pd.read_csv('Twitter_Data.csv')
df = df.dropna()
```

The structure of the dataframe is as follows, where we have the text of the tweet and the assigned sentiment, where -1 is negative, 0 is neutral, and 1 is positive:


```
                                          clean_text  category 
0  when modi promised “minimum government maximum...      -1.0
1  talk all the nonsense and continue all the dra...       0.0
2  what did just say vote for modi  welcome bjp t...       1.0
3  asking his supporters prefix chowkidar their n...       1.0
4  answer who among these the most powerful world...       1.0
```


After that, we can perform the sentiment analysis for all the tweets in the dataset. Therefore, the code to import TextBlob, create a function to iterate through all the rows, and perform sentiment analysis for all the tweets is as follows:

```

from textblob import TextBlob


def analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity

df['new_review_textblop'] = df['clean_text'].apply(analysis)

```

And with this, the dataframe now has a new column, "new_sentiment_textblob," which was:

```
                                               clean_text  ...  new_review_textblop
0       when modi promised “minimum government maximum...  ...            -0.300000
1       talk all the nonsense and continue all the dra...  ...             0.000000
2       what did just say vote for modi  welcome bjp t...  ...             0.483333
3       asking his supporters prefix chowkidar their n...  ...             0.150000
4       answer who among these the most powerful world...  ...             0.400000
```

However, the values are in float and need additional processing so that we have -1, 0, or 1, which indicate negative, neutral, or positive sentiment. The rule is as follows: values less than -0.2 are considered negative sentiment, values between -0.2 and 0.2 are neutral sentiment, and values above 0.2 are positive sentiment.
```
def score_analysis(score):
  if score >= .2:
    return 1
  elif score < .2 and score > -.2:
    return 0
  return -1

df['new_sentiment_textblop'] = df['new_review_textblop'].apply(score_analysis)
```

Now we have a fourth column in the dataframe, where we have only 3 values indicating the sentiment assigned to each tweet:

```
                                               clean_text  ...  new_sentiment_textblop
0       when modi promised “minimum government maximum...  ...                      -1
1       talk all the nonsense and continue all the dra...  ...                       0
2       what did just say vote for modi  welcome bjp t...  ...                       1
3       asking his supporters prefix chowkidar their n...  ...                       0
4       answer who among these the most powerful world...  ...                       1
```

Now we can compare the accuracy of the model using TextBlob with the reference values already included in the dataset. To do this, we will import the accuracy_score from scikit-learn and compare the "category" and "new_sentiment_textblob" columns.

```
from sklearn.metrics import accuracy_score

acuracia_textblop = accuracy_score(df['category'], df['new_sentiment_textblop'])

```

Which returns the value:
```
0.7055329541201087
```

### 4 - Results and Conclusions

Therefore, the model has high accuracy in relation to the reference values in the dataset. In other words, despite being simple, we can see that with the use of just a few libraries in Python, we are able to process thousands of texts and perform an accurate sentiment analysis automatically.

There are some improvements using NLP that could be useful for evaluating a large amount of content, such as searching for key terms related to a specific topic and performing subjectivity analysis to assess whether the content is biased or not.

### 5 - Some Interesting Links

The documentation of [textblob](https://textblob.readthedocs.io/en/dev/#) is an excellent resource for learning more about NLP. Additionally, platforms like [Kaggle](www.kaggle.com) offer a wide variety of datasets for study and training.

As an addition that doesn't fit within the scope of this text, here are three recommendations for articles that discuss the use and development of NLP in great detail:
- [What is sentiment analysis?](https://www.ibm.com/topics/sentiment-analysis)
- [What is Sentiment Analysis? - AWS](https://aws.amazon.com/what-is/sentiment-analysis/#:~:text=Sentiment%20analysis%20is%20the%20process,social%20media%20comments%2C%20and%20reviews.)
- [Sentiment Analysis in Python-81% accuracy](https://medium.com/@liangnguyen612/sentiment-analysis-in-python-81-accuracy-ab5d694b7ef8)
