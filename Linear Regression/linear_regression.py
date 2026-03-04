
import numpy as np
class linear_regression():
    def __init__(self , learning_rate = 0.01 , n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None 

    def fit(self , X , Y):
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            y_pred =np.dot(X,self.weights) + self.bias # For your knowledge we use numpy for vectorization and to make our code faster

            dw =  (1/n_samples) * np.dot(X.T , (y_pred - Y))
            db = (1/n_samples) * np.sum(y_pred - Y) 

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self , x):
        y_predicted = np.dot(x , self.weights) + self.bias
        return y_predicted