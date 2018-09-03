# --Sentiment Prediction using Natural Language Processing--

# 1. Data Preprocessing--

# Importing the libraries
import numpy as np
#For Numerical calculations

import matplotlib.pyplot as plt
# For Data Vizualization

import pandas as pd
# For Data Management

import re
import nltk
# For implementing NLP

from sklearn.feature_extraction.text import CountVectorizer
# For adding the features

from sklearn.cross_validation import train_test_split
# For making Train Test Split

from sklearn.naive_bayes import GaussianNB
# For making Last predictions

from sklearn.metrics import confusion_matrix
# For making Predictons Metrix

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# 2. Cleaning the texts--

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# 3. Creating the Bag of Words model--

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# 4. Splitting the dataset into the Training set and Test set--

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# 5. Fitting Naive Bayes to the Training set--

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# 6. Predicting the Test set results--

y_pred = classifier.predict(X_test)

# 7. Making the Confusion Matrix--

cm = confusion_matrix(y_test, y_pred)