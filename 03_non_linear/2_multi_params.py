from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange, shape
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
from featureNormalize import *
from computeCost_gradientDescent import *
from map_feature import *

import os
# os.chdir("C:/Users/W8-64/Dropbox/TTTN/python/code/Implement/non_linear")

data = loadtxt('ex1data2.txt', delimiter=',')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax.scatter(xs, ys, zs, c='r', marker='x')

ax.set_xlabel('Size of the House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of the House')
plt.show()

feature_count = shape(data)[1]-1
m = shape(data)[0]
X = data[:,0:feature_count]
y = data[:,feature_count]
degree = 2
n=X.shape[1]
elementsCount=round(1+n*degree+n*(n-1)*degree*degree/2)
X_mapped = mapFeatureMultiParam(X,degree)
X_scaled = featureNormalize(X_mapped)
theta = zeros(shape=(elementsCount, 1))
alpha = 0.01
num_iters = 1000
lamb =0
#print(X_scaled)
J_history,theta = gradientDescent(X_scaled,y,theta,alpha,lamb,num_iters)
#print(J_history)
plot(arange(num_iters), J_history, '-r')
xlabel('Number of iterators')
ylabel('Cost function each iterator')
show()