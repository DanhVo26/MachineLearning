from numpy import exp, transpose, log, zeros, ones, where, array, append, column_stack, argmax
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs

def displayData(X, example_width=None):
  plt.close()
  plt.figure()
  if X.ndim == 1:
    X = np.reshape(X, (-1,X.shape[0]))
  # Set example_width automatically if not passed in
  if not example_width or not 'example_width' in locals():
    example_width = int(round(math.sqrt(X.shape[1])))
  plt.set_cmap("gray")
  m, n = X.shape
  example_height = int(n / example_width)
  # Compute number of items to display
  display_rows = int(math.floor(math.sqrt(m)))
  display_cols = int(math.ceil(m / display_rows))
  # Between images padding
  pad = 1
  # Setup blank display
  display_array = -np.ones((pad + display_rows * (example_height + pad),  pad + display_cols * (example_width + pad)))
  # Copy each example into a patch on the display array
  curr_ex = 1
  for j in range(1,display_rows+1):
    for i in range (1,display_cols+1):
      if curr_ex > m:
        break
      # Copy the patch
      # Get the max value of the patch to normalize all examples
      max_val = max(abs(X[curr_ex-1, :]))
      rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
      cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))
      display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
      curr_ex += 1
  
    if curr_ex > m:
      break

  # Display Image
  h = plt.imshow(display_array, vmin=-1, vmax=1)

  # Do not show axis
  plt.axis('off')

  plt.show(block=False)

  return h, display_array

def computeCostReg(theta, X, y, lamb):
  h = sigmoid(X.dot(theta)).flatten()
  m = X.shape[0]
  J = 1/m*(-transpose(y).dot(log(h)) - (1-y).dot(log(1 - h))) + lamb/(2*m)*sum(theta[1:]**2)
  return J

def computeGradReg(theta, X, y, lamb):
  h = sigmoid(X.dot(theta)).flatten()
  m = X.shape[0]
  grad = zeros(shape=(X.shape[1], 1))

  grad[0] = transpose(X[:,0]).dot(h-y)/m
  for i in range(1,X.shape[1]):
    grad[i] = transpose(X[:,i]).dot(h-y)/m + lamb/m*theta[i]
  return grad.flatten()

def decorated_cost_reg(X, y, lamb):
  m,n = X.shape
  X_mapped = ones(shape=(m,n+1))
  X_mapped[:,1:] = X
  def f(theta):
    return computeCostReg(theta, X_mapped, y, lamb)
  def fprime(theta):
    return computeGradReg(theta, X_mapped, y, lamb)
  #Initialize theta parameters
  theta = zeros(X_mapped.shape[1])
  return fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)

def sigmoid(X):
  g = 1/(1 + exp(-X))
  return g

def lrCostFunction(theta, X, y, lamb):
  h = sigmoid(X.dot(theta)).flatten()
  m = X.shape[0]
  J = 1/m*(-transpose(y).dot(log(h)) - (1-y).dot(log(1 - h))) + lamb/(2*m)*sum(theta[1:]**2)

  grad = zeros(shape=(X.shape[1], 1))

  grad[0] = transpose(X[:,0]).dot(h-y)/m
  for i in range(1,X.shape[1]):
    grad[i] = transpose(X[:,i]).dot(h-y)/m + lamb/m*theta[i]

  return J

def predictNN(X,theta1,theta2):
  m,n = X.shape
  X = column_stack((ones(shape=(m,1)),X))
  a2 = sigmoid(X.dot(transpose(theta1)))
  a2 = column_stack((ones(shape=(m,1)),a2))

  a3 = sigmoid(a2.dot(transpose(theta2)))

  p = argmax(a3, axis=1)
  # print(a3)
  # print(a3.shape)
  return p+1 #python start with 0