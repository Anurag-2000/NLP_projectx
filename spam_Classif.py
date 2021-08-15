import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import numpy as np

df = pd.read_csv('Resources\SMSSpamCollection', sep='\t', names=['label','message'])

corpus = []
ps = PorterStemmer()

for i in range(len(df)):
    reviews = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    reviews = reviews.lower().split()
    reviews = [ps.stem(word) for word in reviews if word not in stopwords.words('english') ]
    reviews = ' '.join(reviews)
    corpus.append(reviews)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['label'], drop_first=True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.naive_bayes import MultinomialNB
MB = MultinomialNB()
MB.fit(X_train,np.ravel(y_train,order='C'))

y_pred = MB.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = MB, X = X_train, y = np.ravel(y_train,order='C'), cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


