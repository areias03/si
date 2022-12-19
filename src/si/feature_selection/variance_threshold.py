import sys
sys.path.append('.')
sys.path.append("/Users/alexandre/Documents/Mestrado/2º\ Ano/SIB/si/src/si/data")

from data.dataset import Dataset
import numpy as np
import pandas as pd

class VarianceThreshold:
    
    def __init__(self, threshold: float = 0.0) -> None:
        """
        Variance Threshold feature selection.
        It removes all features whose variance surpass the provided threshold.
        Args:
            threshold (float): Non-negative threshold given by the user. Defaults to 0.
        """
        if threshold < 0:
            raise ValueError("Your threshold must be a non-negative integer.")
        self.threshold = threshold
        self.variance = None
        
    def fit(self, dataset: Dataset) -> "VarianceThreshold":
        """
        Calculates the variance of each feature in a dataset.
        Args:
            dataset (_type_): Dataset object
        """
        X = dataset
        variance = X.get_variance()
        self.variance = variance
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Selects all features with a variance higher than the threshold and returns a new dataset 
        containing only the selected features.
        Args:
            dataset (_type_): dataset object
        """
        X = dataset.X
        
        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=None)
        
    def fit_transform(self, dataset: Dataset) -> None:
        """
        Method to run the fit and transform methods automatically by the user.
        Args:
            dataset (Dataset): Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)
        

if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    a = VarianceThreshold()
    a = a.fit_transform(dataset)
    print(a.print_dataframe())
    print(a.features)
