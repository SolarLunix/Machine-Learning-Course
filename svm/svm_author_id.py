#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print "\n-----------------\n"

#########################################################
### your code goes here ###

# Reduce training size as per instructions
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# Initialise SVM
trainer = SVC(C=10000, kernel="rbf")

# Train the trainer
t0 = time()
trainer.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

# Predict the result of new data
t0 = time()
pred = trainer.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s"

print "\n-----------------\n"

# Determine how accurate the prediction is
score = accuracy_score(labels_test, pred)

# Display prediction score
print "Accuracy:", score

print "\n-----------------\n"

# Check specific results.

test10 = (pred[10] == labels_test[10])
test26 = (pred[26] == labels_test[26])
test50 = (pred[50] == labels_test[50])

print "Test 10 is:", test10, pred[10]
print "Test 26 is:", test26, pred[26]
print "Test 50 is:", test50, pred[50]

# See how many results are for Chris

numChris = 0
for result in pred:
    if result == 1:
        numChris += 1

print numChris
#########################################################


