#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print "-----------------"


#########################################################
### your code goes here ###

# Initialise Trainer
trainer = DecisionTreeClassifier(min_samples_split=40)

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
print "Accuracy:", score

features = len(features_train[0])
print "Features:", features
#########################################################


