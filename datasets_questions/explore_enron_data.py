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

print "\n-----------------------------\n"

# Print out someone's data so it is easily seen what features can be worked with
for feature in enron_data.get("METTS MARK"):
    print feature

print "\n-----------------------------\n"

# Variables to increment
num_poi = 0  # Number of persons of interest available to us
num_salaries = 0  # Number of people's salaries available to us
num_defined_emails = 0  # Number of people's emails we have
num_total_pay = 0  # Number of people with total payments defined
num_total_pay_poi = 0  # Number of pois with total payments defined

# Iterate through the whole list
for person in enron_data:
    # Check Persons of interest
    if enron_data.get(person).get("poi") == 1:
        num_poi += 1
        if enron_data.get(person).get("total_payments") != "NaN":
            num_total_pay_poi += 1

    # Check Salaries
    if enron_data.get(person).get("salary") != "NaN":
        num_salaries += 1

    # Check Email Addresses
    if enron_data.get(person).get("email_address") != "NaN":
        num_defined_emails += 1

    # Check Total Payments
    if enron_data.get(person).get("total_payments") != "NaN":
        num_total_pay += 1

    # Check data for individual people by looking at their last name
    name = person.split(" ")
    if name[0] == "PRENTICE":
        print "Prentice stock:", enron_data.get(person).get("total_stock_value")
    elif name[0] == "COLWELL":
        print "Colwell Emails:", enron_data.get(person).get("from_this_person_to_poi")
    elif name[0] == "SKILLING":
        print "Skilling stock options:", enron_data.get(person).get("exercised_stock_options")
        print "Skilling Total Payments:", enron_data.get(person).get("total_payments")
    elif name[0] == "LAY":
        print "Lay Total Payments:", enron_data.get(person).get("total_payments")
    elif name[0] == "FASTOW":
        print "Fastow Total Payments:", enron_data.get(person).get("total_payments")

print "\n-----------------------------\n"
print "Data Tallies:\n"
print "People in list:", len(enron_data)
print "Features per person:", len(enron_data.get("METTS MARK"))
print "Persons of Interest:", num_poi
print "Number of defined emails:", num_defined_emails
print "Number of defined salaries:", num_salaries
print "Number of undefined total payments:", (len(enron_data) - num_total_pay)
print "Number of poi without total payments undefined:", (num_poi - num_total_pay_poi)


print "\n-----------------------------\n"
