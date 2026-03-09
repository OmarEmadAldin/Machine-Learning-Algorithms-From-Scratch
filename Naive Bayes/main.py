from sklearn import datasets
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayes
import numpy as np

x , y = datasets.make_classification(n_samples=100 , n_features=10 ,n_classes=2 , random_state=1234)
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size=0.2 , random_state=46)

cls = NaiveBayes()
cls.fit(x_train,y_train)
predictions = cls.predict(x_test)

accuracy = np.sum(y_test == predictions) / len(y_test)
print("Naive Bayes Classifier Accuracy" , accuracy)
