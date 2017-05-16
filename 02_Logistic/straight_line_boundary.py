from numpy import loadtxt, where, array, zeros, ones, transpose
from pylab import scatter, show, legend, xlabel, ylabel, plot
from sigmoid import *
# import scipy.optimize as opt
import os

from numpy import exp, transpose, log, zeros



# os.chdir("C:/Users/DanhVo/Dropbox/TTTN/python/code/Data/Logistic")
data = loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:data.shape[1]-1]
y = data[:, data.shape[1]-1]

m = data.shape[0]

pos = where(y==1);
neg = where(y==0);

scatter(X[pos][:,0], X[pos][:,1], c='b', marker='+')
scatter(X[neg][:,0], X[neg][:,1], c='r', marker='x')
xlabel('Score Exam 1')
ylabel('Score Exam 2')
legend(['Admitted', 'Denied'])
# show()

X_1 = ones(shape=(m,data.shape[1]))
X_1[:,1:] = X

theta = decorated_cost(X_1,y)
print(theta)

plot_x = array([min(X_1[:, 1]) - 2, max(X_1[:, 2]) + 2])
# decision boundary : theta0 + theta1*x1 + theta2*x2 >= 0 
#                     => x2 = (-theta0 - theta1*x1)/theta2
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plot(plot_x, plot_y)
legend(['Decision Boundary', 'Not admitted', 'Admitted'])

show()
# draw decision boundary -> how can we draw plot_y ???
p = predict(X_1, theta)
print('Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0))



