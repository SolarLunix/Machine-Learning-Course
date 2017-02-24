#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
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

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)

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
