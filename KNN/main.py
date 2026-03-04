import numpy as np
from Knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from helper_function import plot_decision_boundary
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
iris = datasets.load_iris()

X = iris.data
y = iris.target
print (X.shape)
print (y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.title('Iris Dataset')
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.show()

classifier = KNN(k=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'Accuracy: {accuracy:.2f}')

