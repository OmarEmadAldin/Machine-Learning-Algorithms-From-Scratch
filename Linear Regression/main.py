import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linear_regression import linear_regression
import matplotlib.pyplot as plt
from helper_functions import mean_square_error
 
x , y = datasets.make_regression(n_samples = 100 , n_features = 1 , noise = 20 , random_state = 42)
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 1234)

fig = plt.figure(figsize = (8 , 6))
plt.scatter(x[: , 0] , y , color = 'b' , marker = 'o' , s = 100)
plt.xlabel('x' , fontsize = 20)
plt.ylabel('y' , fontsize = 20)
plt.show()

regression_model = linear_regression()
regression_model.fit(x_train , y_train)
predicted = regression_model.predict(x_test)

mse = mean_square_error(y_test , predicted)
print('Mean Square Error : ' , mse)

y_predicted = regression_model.predict(x)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize = (8 , 6))
m1 = plt.scatter(x_train , y_train , color = cmap(0.9) , s = 100)
m2 = plt.scatter(x_test , y_test , color = cmap(0.5)  , s = 100)
plt.plot(x , y_predicted , color = 'black' , linewidth = 2 , label = 'Prediction')
plt.show()