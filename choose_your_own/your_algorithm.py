#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary]

def picture(classifier, imgName):
    try:
        prettyPicture(classifier, features_test, labels_test, imgName)
    except NameError:
        pass
    

def kNN():
    # Train k Nearest Neighbour
    print "Performing kNN"
    t0 = time()
    clf_knn = KNeighborsClassifier(n_neighbors=4)
    clf_knn.fit(features_train, labels_train)
    print "Training time:", round(time() - t0, 3), "s"

    # Test k Nearest Neighbour
    t0 = time()
    knn_prediction = clf_knn.predict(features_test)
    print "Prediction time:", round(time() - t0, 3), "s"

    # View Accuracy
    knn_acc = accuracy_score(labels_test, knn_prediction)
    print "Accuracy:", knn_acc

    picture(clf_knn, "knnTest.png")


def randomForest():
    # Train the classifier
    print "Perfomring Random Forest"
    t0 = time()
    clf_rnd_forest = RandomForestClassifier()
    clf_rnd_forest.fit(features_train, labels_train)
    print "Training time:", round(time() - t0, 3), "s"

    # Test k Nearest Neighbour
    t0 = time()
    rnd_forest_prediction = clf_rnd_forest.predict(features_test)
    print "Prediction time:", round(time() - t0, 3), "s"

    # View Accuracy
    knn_acc = accuracy_score(labels_test, rnd_forest_prediction)
    print "Accuracy:", knn_acc

    picture(clf_rnd_forest, "rndforest.png")

def adaboost():
    # Train the classifier
    print "Perfomring Adaboost"
    t0 = time()
    clf_ada = AdaBoostClassifier()
    clf_ada.fit(features_train, labels_train)
    print "Training time:", round(time() - t0, 3), "s"

    # Test k Nearest Neighbour
    t0 = time()
    ada_prediction = clf_ada.predict(features_test)
    print "Prediction time:", round(time() - t0, 3), "s"

    # View Accuracy
    knn_acc = accuracy_score(labels_test, ada_prediction)
    print "Accuracy:", knn_acc

    picture(clf_ada, "ada.png")


# Run Machine Learning Methods
kNN()
print "-------------------------------"
randomForest()
print "-------------------------------"
adaboost()
