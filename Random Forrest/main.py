from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from random_forest import RandomForrest

data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForrest(n_trees=10, max_depth=5, min_samples=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")
