from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
import os
os.chdir("C:/Users/DanhVo/Dropbox/TTTN/python/code/Data/Linear")


def feature_normalize(X):
  mean_r = []
  std_r = []

  X_norm = X

  n_c = X.shape[1]
  for i in range(n_c):
    m = mean(X[:, i])
    s = std(X[:, i])
    mean_r.append(m)
    std_r.append(s)
    X_norm[:, i] = (X_norm[:, i] - m) / s
  return X_norm, mean_r, std_r


def compute_cost(X, y, theta):
  '''
  Comput cost for linear regression
  '''
  
  m = y.size

  predictions = X.dot(theta)

  sqErrors = (predictions - y)

  J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

  return J


def gradient_descent(X, y, theta, alpha, num_iters):
  m = y.size
  J_history = zeros(shape=(num_iters, 1))

  for i in range(num_iters):

      predictions = X.dot(theta)

      theta_size = theta.size

      for it in range(theta_size):

          temp = X[:, it]
          temp.shape = (m, 1)

          errors_x1 = (predictions - y) * temp

          theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

      print(theta)
      J_history[i, 0] = compute_cost(X, y, theta)

  return theta, J_history


data = loadtxt('ex1data2.txt', delimiter=',')




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Size of the House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of the House')

plt.show()



X = data[:, :2]
y = data[:, 2]



m = y.size

y.shape = (m, 1)


x, mean_r, std_r = feature_normalize(X)
# print("test")
# print(X)
# print(mean_r)
# print(std_r)
# print("test")

it = ones(shape=(m, 3))
it[:, 1:3] = x


iterations = 100
alpha = 0.01


theta = zeros(shape=(3, 1))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
print("test")
print(theta)
print("test")

print(theta, J_history)
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()


price = array([1.0,   ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house: %f' % (price))
