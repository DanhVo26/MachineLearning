from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy
from computeCost_gradientDescent import *
import os
# os.chdir("C:/Users/DanhVo/Dropbox/TTTN/python/code/Data/Linear")

data = loadtxt('ex1data1.txt', delimiter=',')

# print(data)

X = data[:,0]
y = data[:,1]
scatter(X,y, marker='o',c='r')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
# show()

m = X.size
theta = zeros(shape=(2, 1))
X_1 = ones(shape=(m,2))
X_1[:,1] = X


print(computeCost(X_1,y,theta))

alpha = 0.01
num_iters = 1500

theta = gradientDescent(X_1,y,theta,alpha,num_iters)
result = X_1.dot(theta)
plot(X,result)
show()

# how we draw this contour?
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)

J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))


for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = computeCost(X_1, y, thetaT)


J_vals = J_vals.T

contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
# show()
