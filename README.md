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

Primeiramente, importamos e utilizamos a biblioteca pandas para ler o csv diretamente da pasta local e armazenar em um dataframe bem estruturado. Também é necessário nessa etapa tratar os NA's presentes na amostra. Para esse caso, decidiu-se por apenas excluir as linhas com informações faltantes. 

```
import pandas as pd 

df = pd.read_csv('Twitter_Data.csv')
df = df.dropna()
```

A estrutura do dataframe fica assim, onde tempos o texto do tweet e qual o sentimento atribuido para ele, onde -1 é negativo, 0 é neutro e 1 é positivo:

```
                                          clean_text  category 
0  when modi promised “minimum government maximum...      -1.0
1  talk all the nonsense and continue all the dra...       0.0
2  what did just say vote for modi  welcome bjp t...       1.0
3  asking his supporters prefix chowkidar their n...       1.0
4  answer who among these the most powerful world...       1.0
```


Após isso, já podemos realizar a análise de sentimentos para todos os tweets do dataset. Assim sendo, o código para importar o textblop, criar uma função para iterar todas as linhas e fazer a análise de sentimentos para todos os tweets, temos: 

```

from textblob import TextBlob


def analise_textblop(texto):
    blob = TextBlob(texto)
    sentiment = blob.sentiment
    return sentiment.polarity

df['new_review_textblop'] = df['clean_text'].apply(analise_textblop)

```

E com isso, o dataframe agora possui uma nova coluna, "new_sentiment_textblop", que foi :

```
                                               clean_text  ...  new_review_textblop
0       when modi promised “minimum government maximum...  ...            -0.300000
1       talk all the nonsense and continue all the dra...  ...             0.000000
2       what did just say vote for modi  welcome bjp t...  ...             0.483333
3       asking his supporters prefix chowkidar their n...  ...             0.150000
4       answer who among these the most powerful world...  ...             0.400000
```

No entanto, os valores estão em float e precisam de um tratamento adicional para que tenhamos -1, 0 ou 1, que indicam sentimento negativo, neutro ou positivo. Fazendo valores menores que -.2 serem sentimentos negativos, entre -.2 e .2 sentimentos neutros e acima de .2 positivos: 

```
def analise2(score):
  if score >= .2:
    return 1
  elif score < .2 and score > -.2:
    return 0
  return -1

def analise2_text_blop(score):
  if score >= .2:
    return 1
  elif score < .2 and score > -.2:
    return 0
  return -1

df['new_sentiment_textblop'] = df['new_review_textblop'].apply(analise2_text_blop)
```

Agora tempos uma quarta coluna no dataframe onde temos apenas 3 valores indicando qual sentimento está atribuido para cada tweet:

```
                                               clean_text  ...  new_sentiment_textblop
0       when modi promised “minimum government maximum...  ...                      -1
1       talk all the nonsense and continue all the dra...  ...                       0
2       what did just say vote for modi  welcome bjp t...  ...                       1
3       asking his supporters prefix chowkidar their n...  ...                       0
4       answer who among these the most powerful world...  ...                       1
```

Agora podemos comparar a acurácia do modelo utilizando o textblob com os valores de referência já inclusos no dataset. Para isso, vamos importar o accuracy_score do sci-kit learn e comparar as colunas "category" e "new_sentiment_textblop"

```
from sklearn.metrics import accuracy_score

acuracia_textblop = accuracy_score(df['category'], df['new_sentiment_textblop'])

```

Que retorna o valor: 

```
0.7055329541201087
```

### 4 - Results and Conclusions

Sendo assim, o modelo possui alta acurácia em relação aos valores de referência do dataset. Ou seja, apesar de ser simples, vemos que com o uso de poucas bibliotecas utilizando Python, conseguimos processar milhares de textos e fazer uma análise de sentimento bem acertiva automaticamente. 

Há algumas melhorias utilizando NLP que podem ser úteis para avaliar uma grande quantidade de conteúdo, tal como procurar termos chaves para um determinado assunto e fazer análise de subjetividade para avaliar se os são tendenciosos ou não. 

### 5 - Some Interest Links

A documentação do [textblop](https://textblob.readthedocs.io/en/dev/#) é uma excelente fonte para aprender um pouco mais sobre NLP. Além disso, plataformas como o [Kaggle](kaggle.com) oferecem uma grande variedade de dados para estudo e treinamento. 

Como adicional que não cabem no escopo desse texto, deixo aqui a recomendação de 3 textos que discutem muito bem a utilização e construção de NLP: 
- [What is sentiment analysis?](https://www.ibm.com/topics/sentiment-analysis)
- [What is Sentiment Analysis? - AWS](https://aws.amazon.com/what-is/sentiment-analysis/#:~:text=Sentiment%20analysis%20is%20the%20process,social%20media%20comments%2C%20and%20reviews.)
- [Sentiment Analysis in Python-81% accuracy](https://medium.com/@liangnguyen612/sentiment-analysis-in-python-81-accuracy-ab5d694b7ef8)

