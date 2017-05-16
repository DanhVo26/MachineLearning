from numpy import loadtxt,exp, transpose, log, zeros, where, ones, array, append ,mean ,std
from pylab import scatter, show, legend, xlabel, ylabel, plot
def featureNormalize(X):
  m=X.shape[0]
  n=X.shape[1]
  X_scaled = ones(shape=(m, n))
  for i in range(1,n):
    mean_r = mean(X[:,i])
    std_r = std(X[:,i]) 
    X_scaled[:,i] = (X[:,i] - mean_r)/std_r
  return X_scaled