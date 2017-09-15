##Review Classifier
##It classifies a given review whether it's a positive or negative review.
##It uses Naive Bayes formula to implement the classifying algorithm.
##The training and testing data is given by 10601 course instructor.
##Author: Peter Ye
##Date: 9/14/2017


import os
import csv
import numpy as np
import NB
from NB import NB_YPrior
from NB import logProd
from NB import NB_XGivenY
from NB import NB_Classify
from NB import classificationError

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# TODO: Test logProd function, defined in NB.py

# TODO: Test NB_XGivenY function, defined in NB.py
beta_0 = 50
beta_1 = 70
D = NB_XGivenY(XTrain, yTrain, beta_0, beta_1)
#D = NB_XGivenY(XTrainSmall, yTrainSmall, beta_0, beta_1)

# TODO: Test NB_YPrior function, defined in NB.py
#p = NB_YPrior(yTrain)
p = NB_YPrior(yTrainSmall)
print p

# TODO: Test NB_Classify function, defined in NB.py
yHat = NB_Classify(D, p, XTest)
#print yHat

# TODO: Test classificationError function, defined in NB.py
error = classificationError(yHat, yTest)
print error
# TODO: Run experiments outlined in HW2 PDF












