import math
import numpy as np

beta_1 = 3
beta_0 = 2

D= np.zeros([2,3])
XTrain = np.array([[1,0,1],[0,1,1],[1,1,0],[0,0,1],[0,0,0]])
yTrain = np.array([1,0,1,0,1])

XTest = np.array([[1,0,1],[0,1,1],[1,1,1],[0,1,1],[0,0,1]])

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
print "XTrain:" 
print XTrain
print "XTest:"
print XTest
print 'D:'
print D

y1 = np.count_nonzero(yTrain)
p = 1 - float(y1)/(yTrain.shape[0])
#p = np.log10(p)
print "p:"
print p

def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float

    log_product = np.sum(x)
    return log_product

y_new = np.zeros([2,(XTest.shape[1]+1)])
yHat = np.ones(XTest.shape[0])

print "yHat:"
print yHat
print "y_new shape"
print y_new.shape

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
#        print "counter:"
#        print counter
    print "y_new before"
    print y_new
    y_new = np.log10(y_new)
    row1 = logProd(y_new[0,:])
    row2 = logProd(y_new[1,:])
    if (row1 >= row2):
        yHat[counterm] = 0
    else:
        yHat[counterm] = 1
    counterm += 1
    counter = 1
    print "row1 "
    print row1
##    print "m:"
##    print m
##
print "yHat:"
print yHat

y_Truth = np.ones([1,XTest.shape[0]])
counterF = 0
counterH = 0

for m in np.nditer(y_Truth):
    if (m != yHat[counterH]):
        counterF += 1
    counterH += 1

frac = float(counterF)/yHat.size
print "frac:"
print frac
            
        








        
