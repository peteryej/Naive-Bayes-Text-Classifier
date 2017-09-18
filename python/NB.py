import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float

	log_product = np.sum(x)
	return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	
	## Outputs ##
	# D - (2 by V) numpy ndarray
	D = np.zeros([2, XTrain.shape[1]])
	v = 0
	n = 0
	while ( v < XTrain.shape[1]):
		while ( n < XTrain.shape[0]):
			if ((yTrain[n]==0) and (XTrain[n,v]==1)):
				D[0,v] += 1
			if ((yTrain[n]==1) and (XTrain[n,v]==1)):
				D[1,v] += 1	
			n += 1
		v += 1
		n = 0
        y1 = np.count_nonzero(yTrain)
        y0 = yTrain.shape[0] - y1
	D[0,:] = ((D[0,:]+beta_0-1)/(y0+beta_1+beta_0-2))
	D[1,:] = ((D[1,:]+beta_0-1)/(y1+beta_1+beta_0-2))
	return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float
	y1 = np.count_nonzero(yTrain)
	p = 1 - float(y1)/(yTrain.shape[0])
	return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m
	yHat = np.ones(XTest.shape[0])	
	y_new = np.zeros([2,(XTest.shape[1]+1)])
	counter = 1
	counterm = 0
	for m in np.nditer(yHat):
	    y_new[0, 0] = p
	    y_new[1, 0] = 1-p
	    for i in XTest[counterm,:]:       
	        if (i == 0) :
	            y_new[0, counter] = 1 - D[0,counter-1]
	            y_new[1, counter] = 1 - D[1,counter-1]
	        else:
	            y_new[0, counter] = D[0,counter-1]
	            y_new[1, counter] = D[1,counter-1]
	            
	        counter += 1
	    y_new = np.log10(y_new)
	    row1 = logProd(y_new[0,:])
	    row2 = logProd(y_new[1,:])
	    if (row1 >= row2):
	        yHat[counterm] = 0
	    else:
	        yHat[counterm] = 1
	    counterm += 1
	    counter = 1       

	return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float
	counterF = 0
	counterH = 0

	for m in np.nditer(yTruth):
	    if (m != yHat[counterH]):
	        counterF += 1
	    counterH += 1


	error = float(counterF)/yHat.size
	return error


















