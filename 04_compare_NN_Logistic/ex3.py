from numpy import shape,random,array,zeros,ones
from pylab import show
import allFunction as allFunc
import scipy.io
import numpy as np
np.set_printoptions(threshold=np.inf)


mat = scipy.io.loadmat('ex3data1.mat')

X = mat["X"]
y = mat["y"].flatten()
m,n = X.shape
rand_indices = random.permutation(m)
sel = X[rand_indices[:100],:]

# allFunc.displayData(sel)
# show()

# X_t = array([[1,0.1,0.6,1.1],[1,0.2,0.7,1.2],[1,0.3,0.8,1.3],[1,0.4,0.9,1.4],[1,0.5,1,1.5]])
# y_t = array([1,0,1,0,1])
# lamb_t = 3
# theta_t = array([-2, -1, 1, 2])
# J_t = allFunc.lrCostFunction(theta_t, X_t, y_t, lamb_t)
# grad = allFunc.computeGradReg(theta_t, X_t, y_t, lamb_t)
# print(J_t)
# print(grad)
# # print(type(X_t))
# # print(X_t)
# # print(X_t.shape)
# print(y.shape)
# print(y_t.shape)
# print(n)
lamb = 0.01
theta = zeros(shape=(10,n+1))
num_labels = 10
for i in range(0,num_labels):
	theta[i] = allFunc.decorated_cost_reg(X, (y%10==i).astype(int), lamb)
	pass
m,n = X.shape
X_mapped = ones(shape=(m,n+1))
X_mapped[:,1:] = X
pred = np.argmax(np.dot(X_mapped,theta.T) , axis=1)
# m,n = X.shape
# print(m)
# print(n)
print(theta)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y%10)*100)))
print('Training Set Accuracy for 1:  {:f}'.format(np.mean(pred[500:1000]  == y.flatten()[500:1000]%10)  * 100))
print('Training Set Accuracy for 2:  {:f}'.format(np.mean(pred[1000:1500] == y.flatten()[1000:1500]%10) * 100))
print('Training Set Accuracy for 3:  {:f}'.format(np.mean(pred[1500:2000] == y.flatten()[1500:2000]%10) * 100))
print('Training Set Accuracy for 4:  {:f}'.format(np.mean(pred[2000:2500] == y.flatten()[2000:2500]%10) * 100))
print('Training Set Accuracy for 5:  {:f}'.format(np.mean(pred[2500:3000] == y.flatten()[2500:3000]%10) * 100))
print('Training Set Accuracy for 6:  {:f}'.format(np.mean(pred[3000:3500] == y.flatten()[3000:3500]%10) * 100))
print('Training Set Accuracy for 7:  {:f}'.format(np.mean(pred[3500:4000] == y.flatten()[3500:4000]%10) * 100))
print('Training Set Accuracy for 8:  {:f}'.format(np.mean(pred[4000:4500] == y.flatten()[4000:4500]%10) * 100))
print('Training Set Accuracy for 9:  {:f}'.format(np.mean(pred[4500:5000] == y.flatten()[4500:5000]%10) * 100))
print('Training Set Accuracy for 10: {:f}'.format(np.mean(pred[0:500]     == y.flatten()[0:500]%10)     * 100))