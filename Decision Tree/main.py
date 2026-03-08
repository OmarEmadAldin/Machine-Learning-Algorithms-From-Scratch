from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from decision_tree import DecisionTree
from helper_function import accuracy
data = datasets.load_breast_cancer()
x , y = data.data , data.target
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=10)
clf.fit(x_train , y_train)
predictions = clf.predict(x_test)
acc = accuracy(y_test , predictions)
print("Accuracy:", acc)

