from typing import Callable
import numpy as np
from si.data.dataset import Dataset


class PCA:
    
    def __init__(self, n_components: int) -> None:
        """
        It performs the Principal Component Analysis (PCA) on a given dataset, using the Singular Value Decomposition method.
        Args:
            n_components (int): Number of components to be considered and returned from the analysis.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        
    def fit(self, dataset: Dataset) -> "PCA":
        """
        It fits the data and stores the mean values of each sample, the principial components and the explained variance.
        Args:
            dataset (Dataset): Dataset object.
        """
        # centering data
        self.mean = np.mean(dataset.X, axis = 0)
        self.centered_data = dataset.X - dataset.X.mean(axis=0, keepdims=True)
        
        # calculating SVD
        U, S, Vt = np.linalg.svd(self.centered_data, full_matrices=False)
        
        # principal components
        self.components = Vt[:self.n_components]
        
        # explained variance
        n = len(dataset.X)
        EV = (S ** 2) / (n - 1)
        self.explained_variance = EV[:self.n_components]
        return self    
        
        
    def transform(self, dataset: Dataset) -> None:
        """
        Returns the calculated reduced Singular Value Decomposition (SVD)
        Args:
            dataset (Dataset): Dataset object
        """
        V = self.components.T
        X_reduced = np.dot(self.centered_data, V)
        return X_reduced
    
    def fit_transform(self, dataset: Dataset) -> None:
        """
        It fit and transform the dataset
        Args:
            dataset (Dataset): Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset=dataset)
