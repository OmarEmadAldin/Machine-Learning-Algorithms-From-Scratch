from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from svm import SVM
import numpy as np
from helper_function import visualize_svm

X, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = np.sum (y_test == predictions) / len(y_test)
print("SVM classification accuracy", accuracy)

visualize_svm(X,y , clf)