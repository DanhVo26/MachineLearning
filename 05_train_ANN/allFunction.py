from numpy import exp, transpose, log, zeros, ones, where, array, append, column_stack, argmax, dot 
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

def sigmoid(X):
  g = 1/(1 + exp(-X))
  return g

def sigmoidGradient(X):
  g = sigmoid(X)*(1-sigmoid(X))
  return g

def nnCostfunction(combineTheta, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
  Theta1 = np.reshape(combineTheta[:hidden_layer_size * (input_layer_size + 1)], \
                     (hidden_layer_size, input_layer_size + 1), order='F')

  Theta2 = np.reshape(combineTheta[hidden_layer_size * (input_layer_size + 1):], \
                     (num_labels, hidden_layer_size + 1), order='F')

  a1 = ones(shape=(X.shape[0],X.shape[1]+1))
  a1[:, 1:] = X
  z2 = dot(a1,transpose(Theta1))
  a2 = ones(shape=(z2.shape[0],z2.shape[1]+1))
  a2[:, 1:] = sigmoid(z2)

  z3 = dot(a2,transpose(Theta2))
  a3 = sigmoid(z3)
  h = a3
  m = X.shape[0]
  J = 0

  y_mapped = zeros(shape=(m, num_labels))
  for i in range(0,m):
    y_mapped[i][y[i]-1] = 1
    pass
  J = sum(sum(-y_mapped*log(h) - (1-y_mapped)*log(1-h)))/m
  regularization = (sum(sum(Theta1[:,1:]**2)) + sum(sum(Theta2[:,1:]**2)))/(2*m)*lamb
  J = J + regularization

  # back propagation
  Theta1_grad = zeros(shape=(Theta1.shape))
  Theta2_grad = zeros(shape=(Theta2.shape))
  for t in range(0,m):
    a1 = ones(shape=(1,X.shape[1]+1))
    a1[:,1:] = X[t,:]
    
    z2 = dot(a1,transpose(Theta1))
    a2 = ones(shape=(1,z2.shape[1]+1))
    a2[:,1:] = sigmoid(z2)

    z3 = dot(a2,transpose(Theta2))
    a3 = sigmoid(z3)

    err_3 = a3 - y_mapped[t]
    err_2 = dot(err_3,Theta2)*sigmoidGradient(column_stack((ones(shape=(1,1)),z2)))
    err_2 = err_2[:,1:]
    
    Theta1_grad = Theta1_grad + dot(transpose(err_2), a1)
    Theta2_grad = Theta2_grad + dot(transpose(err_3), a2)
    pass

  Theta1_grad = Theta1_grad/m + lamb/m*column_stack((zeros(shape=(Theta1.shape[0],1)),Theta1[:,1:]))
  Theta2_grad = Theta2_grad/m + lamb/m*column_stack((zeros(shape=(Theta2.shape[0],1)),Theta2[:,1:]))
  grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))
  return J,grad 

def predictNN(X, theta1,theta2):
  m,n = X.shape
  X = column_stack((ones(shape=(m,1)),X))
  a2 = sigmoid(X.dot(transpose(theta1)))
  a2 = column_stack((ones(shape=(m,1)),a2))

  a3 = sigmoid(a2.dot(transpose(theta2)))

  p = argmax(a3, axis=1)
  # print(a3)
  # print(a3.shape)
  return p+1 #python start with 0