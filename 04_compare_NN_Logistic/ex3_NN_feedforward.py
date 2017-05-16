from numpy import shape,random,array,zeros,ones,column_stack,argmax
from pylab import show
from allFunction import *
import scipy.io
import numpy as np
np.set_printoptions(threshold=np.inf)
############# load data ##################
mat = scipy.io.loadmat('ex3data1.mat')

X = mat["X"]
y = mat["y"].flatten()
m,n = X.shape
rand_indices = random.permutation(m)
sel = X[rand_indices[:100],:]

############# load theta ##################
mat = scipy.io.loadmat('ex3weights.mat')

theta1 = mat["Theta1"]
theta2 = mat["Theta2"]

pred = predictNN(X,theta1,theta2)
# print(pred)
# print(y)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y)*100)))
