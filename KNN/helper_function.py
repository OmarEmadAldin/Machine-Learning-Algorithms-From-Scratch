import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap    

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

def plot_decision_boundary(model, X, y):
    h = 0.02  # step size in mesh

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=(0,255,0))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=(255,255,0), edgecolor='k', s=20)

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'KNN Decision Boundary (k={model.k})')
    plt.show()