import numpy as np
from helper_functions import sigmoid
class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None


    def fit(self, X ,Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.num_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - Y))
            db = (1 / n_samples) * np.sum(y_predicted - Y)
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_predicted_cls)