#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "People in list:", len(enron_data)
print "Features per person", len(enron_data.get("METTS MARK"))

num_poi = 0
for person in enron_data:
    if enron_data.get(person).get("poi") == 1:
        num_poi += 1

    name = person.split(" ")
    if name[0] == "PRENTICE":
        print "Prentice stock:", enron_data.get(person).get("total_stock_value")

print "Persons of Interest:", num_poi

