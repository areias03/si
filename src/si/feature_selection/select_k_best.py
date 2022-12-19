import sys
sys.path.append('.')
sys.path.append("/Users/alexandre/Documents/Mestrado/2º\ Ano/SIB/si/src/si/data")
sys.path.append("/Users/alexandre/Documents/Mestrado/2º\ Ano/SIB/si/src/si/statistics")

import numpy as np

from data.dataset import Dataset
from statistics import f_classification
from typing import Callable

class SelectKBest:
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
    
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest': 
        
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        
        idxs = np.argsort(self.F)[-self.K:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)


