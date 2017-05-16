from numpy import zeros,shape,mean,std
def featureNormalize(X):
  # mean_r = []
  # std_r = []

  # X_norm = X

  # n_c = X.shape[1]
  # for i in range(n_c):
  #   m = mean(X[:, i])
  #   s = std(X[:, i])
  #   mean_r.append(m)
  #   std_r.append(s)
  #   X_norm[:, i] = (X_norm[:, i] - m) / s

  X_norm = X
  m = shape(X_norm)[0]
  feature_count = shape(X_norm)[1]
  mean_r = zeros(shape=(feature_count,1))
  std_r = zeros(shape=(feature_count,1))
  # std = sqrt(mean(abs(x - x.mean())**2)).
  # example a=[1,2,3] 
  # m = a.mean() = 2
  # std = sqrt( ( (1-m)^2 + (2-m)^2 + (3-m)^2 ) / a.size() )
  for i in range(0,feature_count):
    mean_r[i] = mean(X_norm[:,i])
    std_r[i] = std(X_norm[:,i])
    # for j in range(0,m):
    #   X_norm[j,i] = (X_norm[j,i] - mean_r[i])/std_r[i]
    X_norm[:,i] = (X_norm[:,i] - mean_r[i])/std_r[i]
  return X_norm, mean_r, std_r