# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:44:43 2018

@author: Ashish
"""
# importing the library
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import pickle
from collections import defaultdict

# loading dataset

df = pd.read_json("News_Category_Dataset.json", lines = True)

df_new = df.drop(['authors', 'date', 'link'], axis = 1)

category_data = defaultdict(list)

category_list = []

input_size = df.shape[0]


# mapping all similar data belonging from same catagory 
for i in range(0, input_size):
    category_data[df_new['category'][i]].append(df_new['headline'][i] + ' ' + df_new['short_description'][i])
    
for category in category_data:
    category_list.append(category)
    
total_category = len(category_list)

training_data = []

# creating training data
for i in range(0, total_category):
    for item in category_data[category_list[i]]:
        training_data.append({'data' : item, 'flag' : i})
        
training_data = pd.DataFrame(training_data, columns=['data', 'flag'])


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, input_size):
    review = re.sub('[^a-zA-Z]', ' ', training_data['data'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

training_data['data'] = corpus
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8')


# Creating the Bag of Words model
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=100)
clf = MultinomialNB().fit(X_train, y_train)

#SAVE MODEL
pickle.dump(clf, open("nb_model.pkl", "wb"))

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("nb_model.pkl","rb"))

#PREDICTION
predicted = clf.predict(X_test)
result_bayes = pd.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_bayes.to_csv("res_bayes.csv", sep = ',')

#ACCURACY SCORE
score = accuracy_score(y_test, predicted)
