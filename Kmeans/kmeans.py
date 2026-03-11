#Unsupervised ML technique used for clustering
'''
The iterative process
1 - init cluster centers randomly (centroids)
2- repeat until converged
    -- update cluster labels : assign point to the nearest cluster centroid
    -- updater centroids as we set center to the mean of each cluster
'''
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__ (self , k = 5  ,max_iters =100 , plot_steps = False):
        self.k = k
        self.iters = max_iters
        self.plot_steps = plot_steps

        # list of clusters 
        self.clusters = [[] for _ in range(self.k)]
        # mean vector for each cluster
        self.centroids = [] 

    def predict (self,X):
        self.X = X
        self.n_sample , self.n_feature = X.shape

        # First Centroids Init 
        # n_sample to chose any random from that and k are number of centroids and replace used to avoid repeating
        # the below line used to append the centroid with index to each centroid point
        random_sample_idxs = np.random.choice(self.n_sample, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        #Optimize clusters
        for _ in range(self.iters):
            self.clusters = self._create_clusters (self.centroids)

            if self.plot_steps:
                self.plot()
            # calculate the new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(centroids_old,self.centroids):
                break

            if self.plot_steps:
                self.plot()
    
        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self , clusters):
        labels = np.empty(self.n_sample)
        for cluster_idx , cluster in enumerate (clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def _create_clusters(self , centroids):
        clusters = [[] for _ in range(self.k)]

        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)

        return clusters    
    def _closest_centroid(self , sample , centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _get_centroids(self,clusters):
        centroids = np.zeros((self.k , self.n_feature))
        for cluster_idx , cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster] , axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
        
    def _is_converged(self,old_centroids , centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

