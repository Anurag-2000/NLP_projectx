import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re


para = ' '

stemmer = PorterStemmer()
lemmatize = WordNetLemmatizer()
sentence = nltk.sent_tokenize(para)
corpus = []

for i in range(len(sentence)):
    reviews = re.sub('[^a-zA-Z]', ' ', sentence[i])
    reviews = reviews.lower()
    reviews = reviews.split()
    reviews = [lemmatize.lemmatize(word) for word in reviews if word not in set(stopwords.words('english'))]
    reviews = ' '.join(reviews)
    corpus.append(reviews)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()
