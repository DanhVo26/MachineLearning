from numpy import loadtxt, zeros, ones, array, linspace, logspace, random
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np
import os
os.chdir("C:/Users/DanhVo/Dropbox/TTTN/python/code/Data/Linear")

# data = loadtxt('ex1data1.txt', delimiter=',')

# # print(data)
# X = data[:,0]
# y = data[:,1]
# m = X.size
# theta = ones(shape=(2,1))#.flatten()

# plot([1,1],[10,10],'k-',lw=3)
# show()

# print(theta[0])
# X_1 = ones(shape=(m,2))
# X_1[:,1] = X


# print(theta)
# print(X_1[1])
# # h = np.dot(theta,X_1)
# print(X)
# print(data.transpose())
# print(h)

mean = zeros(shape=random(10,10))
# mean[i] = mean(X[:,i])
print(mean)

a = array([[1,2,3]])
print(a)
print(a.shape)
a.shape=(3,1,1)
# [[[]] [[]] [[]]]
print(a)
# print(a.shape= (3,1) )
y = zeros((2, 3, 4))
print(y)
print(y.shape)
y.shape = (3,8)
print(y)
print(y.shape)