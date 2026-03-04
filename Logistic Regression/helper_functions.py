import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) + 1e-15  # Adding a small constant to prevent division by zero

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)