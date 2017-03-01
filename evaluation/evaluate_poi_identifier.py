#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn.model_selection import train_test_split
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=.3, random_state=42)

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
print "Accuracy:", score

features = len(features_train[0])
print "Features:", features

pred = [0] * len(labels_test)
score = accuracy_score(labels_test, pred)

# Display prediction score
print "Accuracy:", score

