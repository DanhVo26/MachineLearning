from numpy import exp, transpose, log, zeros, where, array, append
from pylab import scatter, show, legend, xlabel, ylabel, plot
from scipy.optimize import fmin_bfgs
import numpy

def plotData(X, y):
  pos = where(y==1);
  neg = where(y==0);

  scatter(X[pos][:,0], X[pos][:,1], c='b', marker='+')
  scatter(X[neg][:,0], X[neg][:,1], c='r', marker='x')
  pass

def sigmoid(X):
  g = 1/(1 + exp(-X))
  return g

def computeCost(theta,X, y):
  h = sigmoid(X.dot(theta)).flatten()
  m = X.shape[0]
  J = 1/m*(-transpose(y).dot(log(h)) - (1-y).dot(log(1 - h)))
  return J

def computeGrad(theta,X,y):
  h = sigmoid(X.dot(theta)).flatten()
  m = X.shape[0]
  grad = zeros(shape=(X.shape[1], 1))
  for i in range(0,X.shape[1]):
    grad[i] = transpose(X[:,i]).dot(h-y)/m  
  return grad.flatten()



# ----------------- function for straight line boundary 
def decorated_cost(X_1, y):
  def f(theta):
    return computeCost(theta, X_1, y)
  def fprime(theta):
    return computeGrad(theta, X_1, y)
  #Initialize theta parameters
  theta = zeros(3)
  return fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)
  #fmin unc

def predict(X_1, theta):
  m = X_1.shape[0]
  predict = zeros(shape=(m,1))
  predict = X_1.dot(theta)

  for i in range(0,m):
    # predict[i] >= 0 ? 1:0
    if predict[i] >= 0 :
      predict[i] = 1
    else:
      predict[i] = 0
  return predict
# -----------------------END---------------------------------



# ----------------- function for complex shape boundary 
def mapFeature(X_1, X_2):
  m = X_1.size
  degree = 6
  elementsCount = int((degree+2) * (degree+1)/2)
  out = array([])
  for loop_X in range(0,m):
    t = array([])
    for i in range(0,degree+1):
      for j in range(0,i+1):
        t = append(t, X_1[loop_X]**(i-j) * X_2[loop_X]**j)
      pass
    pass
    out = append(out, t)
  pass
  out.shape = (m,elementsCount)
  return out

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

def decorated_cost_reg(X_mapped, y, lamb):
  def f(theta):
    return computeCostReg(theta, X_mapped, y, lamb)
  def fprime(theta):
    return computeGradReg(theta, X_mapped, y, lamb)
  #Initialize theta parameters
  theta = zeros(X_mapped.shape[1])
  return fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)

# -----------------------END-----------------------------------


