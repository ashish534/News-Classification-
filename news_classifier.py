# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:44:43 2018

@author: Ashish
"""

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
from collections import defaultdict

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_json(r'C:\Users\Ashish\Pictures\News_Category_Dataset.json', lines = True)

df_new = df.drop(['authors', 'date', 'link'], axis = 1)

category_data = defaultdict(list)

category_list = []

input_size = df.shape[0]

for i in range(0, input_size):
    category_data[df_new['category'][i]].append(df_new['headline'][i] + ' ' + df_new['short_description'][i])
    
for category in category_data:
    category_list.append(category)
    
total_category = len(category_list)

training_data = []

for i in range(0, total_category):
    for item in category_data[category_list[i]]:
        training_data.append({'data' : item, 'flag' : i})
        
training_data = pd.DataFrame(training_data, columns=['data', 'flag'])
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8')

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)

docs_new = "Messi joins other football team"
docs_new = [docs_new]

#LOAD MODEL

X_new_counts = X_train_counts.transform(docs_new)
X_new_tfidf = X_train_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print(category_list[predicted[0]])
