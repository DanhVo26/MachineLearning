from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

import os
os.chdir("C:/Users/DanhVo/Dropbox/TTTN/python/code/Data/Linear")
# os.chdir("/home/DanhVo/Dropbox/TTTN/python/code/Data/Linear")


data = loadtxt('ex1data1.txt', delimiter=',')


scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')


X = data[:, 0]
y = data[:, 1]



m = y.size


it = ones(shape=(m, 2))
it[:, 1] = X


theta = zeros(shape=(2, 1))


iterations = 1500
alpha = 0.01

def compute_cost(X, y, theta):
  '''
  Comput cost for linear regression
  '''
  
  m = y.size

  predictions = X.dot(theta).flatten()

  sqErrors = (predictions - y) ** 2

  J = (1.0 / (2 * m)) * sqErrors.sum()

  return J


def gradient_descent(X, y, theta, alpha, num_iters):
  m = y.size
  J_history = zeros(shape=(num_iters, 1))

  for i in range(num_iters):

      predictions = X.dot(theta).flatten()

      errors_x1 = (predictions - y) * X[:, 0]
      errors_x2 = (predictions - y) * X[:, 1]

      theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
      theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

      J_history[i, 0] = compute_cost(X, y, theta)

  return theta, J_history


theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
print("test")
print("test")
print("test")
print("test")
print("test")
print("test")
print(theta)
print(J_history)
print("test")
print("test")
print("test")
print("test")
print("test")
print("test")
predict1 = array([1, 3.5]).dot(theta).flatten()
print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
predict2 = array([1, 7.0]).dot(theta).flatten()
print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))

print('theta',theta)
print('xxxxx', it)
result = it.dot(theta)#.flatten()
print('result',result)
plot(data[:, 0], result)
show()



theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)



J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))


for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(it, y, thetaT)


J_vals = J_vals.T

contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
show()