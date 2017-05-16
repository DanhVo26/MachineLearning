from numpy import shape,random,array,zeros,ones
from pylab import show
from allFunction import *
import scipy.io
import numpy as np
from scipy.optimize import minimize
np.set_printoptions(threshold=np.inf)

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   


mat = scipy.io.loadmat('ex4data1.mat')

X = mat["X"]
y = mat["y"].flatten()
m,n = X.shape
rand_indices = random.permutation(m)
sel = X[rand_indices[:100],:]

displayData(sel)
show()

mat = scipy.io.loadmat('ex4weights.mat')
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

combineTheta = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))
lamb = 1
nnCostfunction(combineTheta, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)

print('Training Neural Network...')

maxiter = 20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
results = minimize(nnCostfunction, x0=combineTheta, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

combineTheta = results["x"]

# Obtain Theta1 and Theta2 back from combineTheta
Theta1 = np.reshape(combineTheta[:hidden_layer_size * (input_layer_size + 1)], \
                 (hidden_layer_size, input_layer_size + 1), order='F')

Theta2 = np.reshape(combineTheta[hidden_layer_size * (input_layer_size + 1):], \
                 (num_labels, hidden_layer_size + 1), order='F')

pred = predictNN(X, Theta1, Theta2)

print('Training Set Accuracy: {:f}'.format( ( np.mean(pred == y)*100 ) ) )

print('Training Set Accuracy for 1:  {:f}'.format(np.mean(pred[500:1000]  == y.flatten()[500:1000]%10)  * 100))
print('Training Set Accuracy for 2:  {:f}'.format(np.mean(pred[1000:1500] == y.flatten()[1000:1500]%10) * 100))
print('Training Set Accuracy for 3:  {:f}'.format(np.mean(pred[1500:2000] == y.flatten()[1500:2000]%10) * 100))
print('Training Set Accuracy for 4:  {:f}'.format(np.mean(pred[2000:2500] == y.flatten()[2000:2500]%10) * 100))
print('Training Set Accuracy for 5:  {:f}'.format(np.mean(pred[2500:3000] == y.flatten()[2500:3000]%10) * 100))
print('Training Set Accuracy for 6:  {:f}'.format(np.mean(pred[3000:3500] == y.flatten()[3000:3500]%10) * 100))
print('Training Set Accuracy for 7:  {:f}'.format(np.mean(pred[3500:4000] == y.flatten()[3500:4000]%10) * 100))
print('Training Set Accuracy for 8:  {:f}'.format(np.mean(pred[4000:4500] == y.flatten()[4000:4500]%10) * 100))
print('Training Set Accuracy for 9:  {:f}'.format(np.mean(pred[4500:5000] == y.flatten()[4500:5000]%10) * 100))
print('Training Set Accuracy for 10: {:f}'.format(np.mean(pred[0:500]     == y.flatten()[0:500])     * 100))

