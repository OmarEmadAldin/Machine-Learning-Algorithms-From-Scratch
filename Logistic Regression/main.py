import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
from helper_functions import accuracy
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

logistic_model = LogisticRegression(learning_rate=0.001, num_iterations=1000)
logistic_model.fit(X_train, Y_train)
predictions = logistic_model.predict(X_test)
accuracy = accuracy(Y_test, predictions)
print("Logistic Regression classification accuracy: ", accuracy)

