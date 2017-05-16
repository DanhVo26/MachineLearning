from numpy import zeros
def computeCost(X, y, theta):
  m = y.size
  
  h = X.dot(theta).flatten() 
  J = sum((h-y)**2)/(2*m);
  return J

def gradientDescent(X, y, theta, alpha,lamb, num_iters):
  temp = zeros(shape=(X.shape[1]))
  feature_count = X.shape[1]
  J_history = zeros(shape=(num_iters,1))
  m = X.shape[0]
  for i in range(num_iters):
    h = X.dot(theta).flatten()
    temp[0] = (X[:,0].dot((h-y)))*alpha/m
    theta[0,0] = theta[0,0] - temp[0]
    for i_temp in range(1,feature_count):
      temp[i_temp] = (X[:,i_temp].dot((h-y)))*alpha/m +theta[i_temp,0]*alpha*lamb/m
      theta[i_temp,0] = theta[i_temp,0] - temp[i_temp]
    J_history[i] = computeCost(X,y,theta)
  return  J_history,theta