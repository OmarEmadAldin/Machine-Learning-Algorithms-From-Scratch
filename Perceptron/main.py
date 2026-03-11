from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from perceptron import Perceptron
import numpy as np
X , Y = datasets.make_blobs(n_samples= 150 , n_features= 2 ,centers=2, cluster_std=1.05, random_state=1234)
X_train ,X_test, y_train , y_test = train_test_split(X , Y , train_size=0.8 , random_state=47)

cls = Perceptron(learning_rate=0.1 , n_iters= 100)
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
acc = np.sum ( y_test == predictions) / len(y_test)
print("The accuracy of the predictions are",acc)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-cls.weights[0] * x0_1 - cls.bias) / cls.weights[1]
x1_2 = (-cls.weights[0] * x0_2 - cls.bias) / cls.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()