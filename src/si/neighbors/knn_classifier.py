from typing import Callable
from si.statistic.euclidean_distance import euclidean_distance
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

import numpy as np


class KNNClassifier:
    
    def __init__(self, k: int, distance: Callable = euclidean_distance) -> None:
        """
        Algorithm that predicts the class for a sample using the k most similar examples.
        Args:
            k (int): number of examples to consider
            distance (Callable, optional): euclidean distance function. Defaults to euclidean_distance.
        """
        self.k = k
        self.distance = distance
        self.dataset = None
        
    def fit(self, dataset: Dataset) -> "KNNClassifier":
        self.dataset = dataset 
        return self
    
    def _get_closest_label(self, sample):
        # Calculates the distance between the samples and the dataset
        distances = self.distance(sample, self.dataset.X)

        # Sort the distances and get indexes
        knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array
        knn_labels = self.dataset.y[knn]

        # Returns the unique classes and the number of occurrences from the matching classes
        labels, counts = np.unique(knn_labels, return_counts=True)

        # Gets the most frequent class
        high_freq_lab = labels[np.argmax(counts)]  # get the indexes of the classes with the highest frequency/count

        return high_freq_lab

    def predict(self, dataset: Dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)        
    
    def score(self, dataset: Dataset) -> float:
        prediction = self.predict(dataset)
        return accuracy(dataset.y, prediction)
