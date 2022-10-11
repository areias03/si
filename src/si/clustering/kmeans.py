import numpy as np

from si.data.dataset import Dataset
from si.statistics import euclidean_distance

class KMeans:
    
    def __init__(self,k: int,max_iter: int=1000,distance: Callable = euclidean_distance):
        self.k= k
        self.max_iter= max_iter
        self.distance= distance

    def _init_centroids(self,dataset: Dataset) -> Dataset:
        seeds = np.random.permutation(dataset.shape()[0][:self.k])
        centroids = dataset.X[seeds]

    def _get_closet_centroid(self,sample:np.array) -> np.array:
        centroids_distance = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distance, axis=0)
        return closest_centroid_index


    def fit(self,dataset: Dataset) -> Dataset:
        
        convergence = False

        labels = np.zeros(dataset.shape(0))

        while not convergence and i < self.max_iter:
                
            new_labels = np.apply_along_axis(self._get_closet_centroid, axis=1,arr=dataset.X)

            centroids = []
            for i in range(self.k):
                centroid = np.mean(dataset.X[new_labels == i],axis =0)
                centroids.append(centroid)
            self.centroids = np.array(labels,)
            

    def _get_distance(self,sample:np.ndarray) -> np.ndarray:

        return


    def transform(self,dataset: Dataset) ->np.ndarray:
        centroids_distances = np.apply_along_axis(self._get_distance, axis = 1, arr=dataset.X)
        return centroids_distances

    def predict(self,dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closet_centroid,axis = 1, arr = dataset.X)