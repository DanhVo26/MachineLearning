from numpy import zeros
def computeCost(X, y, theta):
  m = y.size
  
  h = X.dot(theta).flatten()
  
  J = sum((h-y)**2)/(2*m);
  return J

def gradientDescent(X, y, theta, alpha, num_iters):
  m = y.size
  for i in range(1,num_iters):
    h = X.dot(theta).flatten()
    temp0 = alpha/m*sum(h-y)
    temp1 = alpha/m*(X[:,1].dot((h-y)))
    theta[0] = theta[0] - temp0
    theta[1] = theta[1] - temp1
  return theta


