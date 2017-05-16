from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange, shape
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
from featureNormalize import *
from computeCost_gradientDescent_multi import *

import os
os.chdir("C:/Users/DanhVo/Dropbox/TTTN/python/code/Data/Linear")



data = loadtxt('ex1data2.txt', delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# n = 100
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Size of the House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of the House')

plt.show()

feature_count = shape(data)[1]-1
m = shape(data)[0]
X = data[:,0:feature_count]
y = data[:,feature_count]
# print(X)
x,mu,sigma = featureNormalize(X)

X_1 = ones(shape=(m,feature_count+1))
X_1[:,1:feature_count+1] = x

alpha = 0.01
num_iters = 400
theta = zeros(shape=(shape(data)[1],1))

J_history,theta = gradientDescent(X_1, y, theta, alpha, num_iters)
# print(theta)
# print(J_history)
plot(arange(num_iters), J_history, '-r')
xlabel('Number of iterators')
ylabel('Cost function each iterator')
show()