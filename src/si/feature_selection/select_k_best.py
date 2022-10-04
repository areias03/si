import numpy as np

from si.data.dataset import Dataset
from si.statistics import f_classification

class SelectKBest:
    def __init__(self,scorefn, k):
        self.scorefn = scorefn
        self.k = k
        self.F = None
        self.p = None

    def fit(self,dataset: Dataset): 
        F,p = self.scorefn(dataset)

    def transform(self,dataset: Dataset):
        idxs = np.argsort(self.F)[-self.K:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self,dataset: Dataset):
        pass


