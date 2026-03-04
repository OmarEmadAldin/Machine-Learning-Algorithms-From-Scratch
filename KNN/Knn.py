import numpy as np
from collections import Counter
from helper_function import euclidean_distance
class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
    
    def predict(self, x):
        predictions = [self._predict(x_i) for x_i in x]
        return predictions

    def _predict(self, x):
        #compute distances between x and all examples in the training set
        distances = euclidean_distance(x, self.X_train)
        #sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    