#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from sklearn.naive_bayes import GaussianNB
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Initialise GaussianNB
gaus = GaussianNB()

# Train gaus
t0 = time()
gaus.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

# Predict the result of new data
t0 = time()
pred = gaus.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s"

# Determine how accurate the prediction is
score = accuracy_score(labels_test, pred)

# Display prediction score
print "Accuracy:", score

#########################################################


