from numpy import zeros,shape

def computeCost(X, y, theta):
  m = y.size
  
  h = X.dot(theta).flatten()
  
  J = sum((h-y)**2)/(2*m);
  return J

def gradientDescent(X, y, theta, alpha, num_iters):
  temp = zeros(shape=(shape(X)[1]))
  J_history = zeros(shape=(num_iters,1))
  feature_count = shape(X)[1]
  m = shape(X)[0]
  for i in range(num_iters):
    h = X.dot(theta).flatten()
    for i_temp in range(0,feature_count):
      temp[i_temp] = (X[:,i_temp].dot((h-y)))*alpha/m
      theta[i_temp,0] = theta[i_temp,0] - temp[i_temp]
    J_history[i] = computeCost(X,y,theta)
  # print(J_history)
  return J_history, theta