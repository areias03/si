from cProfile import label
from os import get_terminal_size
from statistics import mean
import numpy as np
import pandas as pd

class Dataset:

    def __init__(self,X, y, features, label):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        "Verifica as dimensões do dataset"
        return self.X.shape

    def has_label(self):
        "Verifica se o dataset tem y"
        if self.y is not None:
            return True

        return False

    def get_classes(self):
        "Devolve as classes do dataset, ou seja, os valores possíveis de y"
        if self.y is None:
            return

        return np.unique(self.y)

    def get_mean(self):
        "Devolve a média (axis=0 é para colunas, axis=1 é para os exemplos)"
        return np.mean(self.X, axis=0)

    def get_variance(self):
        return np.var(self.X, axis=0)

    def get_median(self):
        return np.median(self.X, axis=0)

    def get_min(self):
        return np.min(self.X, axis=0)

    def get_max(self):
        return np.max(self.X, axis=0)

    def summary(self):
        return pd.DataFrame(
            {'mean': self.get_mean(),
            'variance': self.get_variance(),
            'median': self.get_median(),
            'min': self.get_min(),
            'max': self.get_max()}
        )




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

