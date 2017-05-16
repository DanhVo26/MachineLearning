from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace
from pylab import scatter, show, legend, xlabel, ylabel, contour, title
from sigmoid import *
from scipy.optimize import fmin_bfgs

import os
# os.chdir("C:/Users/DanhVo/Dropbox/TTTN/python/code/Data/Logistic")
data = loadtxt('ex2data2.txt', delimiter=',')

X = data[:, 0:data.shape[1]-1]
y = data[:, data.shape[1]-1]
m = data.shape[0]

plotData(X,y)
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend(['y = 1', 'y = 0'])
show()

X_mapped = mapFeature(X[:,0],X[:,1])
n_mapped = X_mapped.shape[1]
theta = zeros(shape=(n_mapped,1))
# try to change lamb to 0, 100, ... -- default = 1
lamb = 0
theta = decorated_cost_reg(X_mapped, y, lamb)
# --------------------------------------------

u = linspace(-1, 1.5, 50)
v = linspace(-1, 1.5, 50)
z = zeros(shape=(len(u), len(v)))
for i in range(0,len(u)):
  for j in range(0,len(v)):
    z[i, j] = mapFeature(array([u[i]]),array([v[j]])).dot(theta)
# transpose z for what?
z = z.T
plotData(X,y)
contour(u, v, z)
title('lambda = %f' % lamb)
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend(['y = 1', 'y = 0', 'Decision boundary'])
show()
