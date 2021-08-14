import pandas as pd
import nltk as nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

df = pd.read_csv('Resources\SMSSpamCollection', sep='\t', names=['label','message'])

corpus = []
ps = PorterStemmer()

for i in range(len(df)):
    reviews = re.sub('[^a-zA-Z]', ' ', df[i])
    reviews = reviews.lower().split()
    reviews = []
    