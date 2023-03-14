from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

fileName = "esteebrandsdata.csv"
column = "TextReview"
Data = pd.read_csv(fileName,encoding="Latin-1")
Data = Data.replace(np.nan,' ',regex=True)
sentences = list(Data[column])

sid = SentimentIntensityAnalyzer()
sentiments = []
for sentence in sentences:
    ss = sid.polarity_scores(sentence)
    sentiments.append(ss)

print(sentiments)