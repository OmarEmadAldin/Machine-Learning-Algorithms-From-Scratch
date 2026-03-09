from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pca import PCA

data = datasets.load_iris()
x = data.data
y = data.target

pca_DR = PCA(2)
pca_DR.fit(x)
projected_res = pca_DR.transform(x)

print("shape of x:" , x.shape)
print("shape of transformed x:" , projected_res)

"For plotting"
x1 = projected_res[: ,0]
x2 = projected_res[: ,1]

plt.scatter(x1,x2, c=y , edgecolors='none' , alpha=0.8 , cmap=plt.cm.get_cmap("viridis",3))
plt.xlabel("Principal Comp 1")
plt.ylabel("Principal Comp 2")
plt.colorbar()
plt.show()