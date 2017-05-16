from numpy import loadtxt,exp, transpose, log, zeros, where, ones, array, append
from pylab import scatter, show, legend, xlabel, ylabel, plot
def mapFeatureOneparam(X,degree):
#map x ->1,x,x^2..x^degree
  m=X.size
  X_1 = ones(shape=(m,degree+1))
  for i in range(1,degree+1):
    X_1[:,i]= X*X_1[:,i-1]
  return X_1
def mapFeatureMultiParam(X,degree):
#map luy thua 1 phan tu va 2 phan tu
  m=X.shape[0] 
  n=X.shape[1] 
  elementsCount=round(1+n*degree+n*(n-1)*degree*degree/2)
  X_1 = ones(shape=(m,elementsCount))
  #luy thua 1 phan tu
  index=1
  for i in range(0,n):
      X_i=X[:,i]
      X_1[:,index]=X_i
      index=index+1
      for j in range(2,degree+1):
        X_1[:,index]= X_i*X_1[:,index-1]
        index=index+1
  #luy thua 2 phan tu
  for i in range(0,n):
    for j in range(i+1,n):
      a= ones(m)
      for d1 in range(1,degree+1):
        a=a*X[:,i]
        b= ones(m)
        for d2 in range(1,degree+1):
        	b=b*X[:,j]
        	X_1[:,index]=a*b
        	index=index+1
  return X_1