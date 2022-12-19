from cProfile import label
from os import get_terminal_size
from statistics import mean
import numpy as np
import pandas as pd

class Dataset:

    def __init__(self, X:np.ndarray = None, y:np.ndarray = None, features:list = None, label:str = None):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> tuple:
        """
        Returns dataset dimensions.
        """
        return self.X.shape

    def has_label(self):
        """
        Returns True if dataset has labels.
        """
        if self.y is not None:
            return True

        return False

    def get_classes(self) -> list[int]:
        """
        Returns classes as a list.
        """        
        if self.y is None:
            return

        return np.unique(self.y)

    def get_mean(self):
        """
        Returns mean of features.
        """
        return np.mean(self.X, axis=0)

    def get_variance(self):
        """
        Returns variance of features.
        """
        return np.var(self.X, axis=0)

    def get_median(self):
        """
        Returns median of features.
        """
        return np.median(self.X, axis=0)

    def get_min(self):
        """
        Returns minimum value of features
        """
        return np.min(self.X, axis=0)

    def get_max(self):
        """
        Returns maximum value of features
        """
        return np.max(self.X, axis=0)

    def summary(self):
        """
        Returns summarized statistics of the dataset
        """
        return pd.DataFrame(
            {'mean': self.get_mean(),
            'variance': self.get_variance(),
            'median': self.get_median(),
            'min': self.get_min(),
            'max': self.get_max()}
        )

    def print_dataframe(self):
        """
        Prints the dataframe in pd.DataFrame format.
        """
        if self.X is None:
            return

        return pd.DataFrame(self.X, columns=self.features, index=self.y)

    def dropna(self):
        """
        Removes samples with at least one NaN value.
        """
        return pd.DataFrame(self.X).dropna(axis=0).reset_index(drop=True)

    def fillna(self,value: int):
        """
        Fills all NaN values with the given value
        """
        return pd.DataFrame(self.X).fillna(value)




if __name__ == '__main__':
    x = np.array([[1,2,3], [1,2,3]])
    y = np.array([1,2])
    features = ['A','B','C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features=features, label=label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.summary())
    print(dataset.print_dataframe())
