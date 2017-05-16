from numpy import loadtxt, zeros, ones, array, linspace, logspace,arange
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from computeCost_gradientDescent import *
from map_feature import *
from featureNormalize import *
import os

#os.chdir("C:/Users/W8-64/Dropbox/TTTN/python/code/Implement/non_linear")
#data = loadtxt('oneparam_complex.txt', delimiter=',') #thu overfitting
data = loadtxt('oneparam_simple.txt', delimiter=',')

X = data[:,0]
y = data[:,1]

scatter(X,y, marker='o',c='r') 
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
#show()
degree = 5
X_mapped = mapFeatureOneparam(X,degree)
#print(X_mapped)
X_scaled = featureNormalize(X_mapped)
theta = zeros(shape=(degree+1, 1))
alpha = 0.01
num_iters = 1000
lamb =0
J_history,theta = gradientDescent(X_scaled,y,theta,alpha,lamb,num_iters)
#print(J_history)
result = X_scaled.dot(theta)
plot(X,result)
show()
plot(arange(num_iters), J_history, '-r')
xlabel('Number of iterators')
ylabel('Cost function each iterator')
show()
