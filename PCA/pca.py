# For calculating it we need to get variance
# Also the covariance matrix which is alike the variance but for two variables 
# Then we need to calc Eigen vectors, Eigen values
# Don't implement PCA Without knowing the math behind

'''
Steps:
- subtract mean from x
- calculate cov(x,x)
- calc eignenvalue and eigenvectors
- sort the eigenvectors according to their eigenvalues in decreasing order
- choose first k eigenvectors and thet will be the new k dimensions
- transform the original n-dimensions data points into the k dimension , projections with dot product
'''
import numpy as np

class PCA():
    def __init__(self , k):
        self.k = k
        self.components = None
        self._mean = None
        self._var = None
        pass
    
    def fit(self,X):
        # Mean Calc
        self._mean = np.mean(X , axis=0)
        X = X - self._mean
        # Covariance calc
        cov = np.cov(X.T) 
        # Eigenvectors,Eigen values
        eigen_vec , eigen_val = np.linalg.eig(cov)
        eigen_vec = eigen_vec.T
        # Sort
        indx = np.argsort(eigen_val)[::-1]
        eigen_val = eigen_val[indx]
        eigen_vec = eigen_vec[indx]

        self.components = eigen_vec[:self.k]
        print("THE PCA Components are",self.components)
        

    def transform(self,X):
        X = X - self._mean
        transorm_proj = np.dot(X,self.components.T)
        return transorm_proj