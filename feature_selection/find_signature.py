#!/usr/bin/python

import pickle
import os
import numpy
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r"))



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]




### your code goes here
# Initialise Trainer
trainer = DecisionTreeClassifier()

# Train the trainer
t0 = time()
trainer.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

# Predict the result of new data
t0 = time()
pred = trainer.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s"

print "-----------------"

# Determine how accurate the prediction is
score = accuracy_score(labels_test, pred)

# Display prediction score
print "Test Accuracy:", score

features = len(features_train[0])
print "Features:", features

print "-----------------"

a_min = True
over_ = []
for i in range(len(trainer.feature_importances_)):
    af = trainer.feature_importances_[i]
    if af > 0.2:
        over_.append(i)
        a_min = False

if a_min:
    print "No features over 0.2"
else:
    for num in over_:
        print vectorizer.get_feature_names()[num]
