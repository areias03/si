from si.data.dataset import Dataset
from si.statistic.f_classification import f_classification
import numpy as np
from typing import Callable


class SelectPercentile:
    """
    Class that filters dataset variables based on their F-scores. Selects all variables
    with F-score values above the specified corresponding percentile.
    """
    
    def __init__(self, score_func:Callable= f_classification, percentile:float = 0.25):
        """
        Stores the input values.
        
        Args
        ----------
        :param score_func (Callable, optional): Scoring function. Defaults to f_classification
        :param percentile (float, optional): Percentile value cut-off. Only F-scores above this
                           value will remain in the filtered dataset. Defaults to 0.25.
        """
        
        if percentile > 1 or percentile < 0:
            raise ValueError("Percentile value must be a float between 0 and 1")

        self.score_func = score_func
        self.percentile = percentile
        
         
        
    def fit(self, dataset:Dataset) -> "SelectPercentile":
        """
        Stores the F-scores and p-values of each variable of the dataset.
        
        Args
        ----------
        :param dataset (Dataset): Dataset object.
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    
    
    def transform(self, dataset:Dataset) -> Dataset:
        """
        Returns a filtered version of the given Dataset instance using their
        F-scores. The new dataset will have only the variables with F-scores above
        the specified percentile value.
        
        Args
        ----------
        :param dataset (Dataset): Dataset object.
        """
        inds = np.argsort(self.F)[::-1]
        ord_vals = np.sort(self.F)[::-1]
        perc_vals = np.percentile(ord_vals, self.percentile)
        
        inds = inds[:sum(ord_vals <= perc_vals)]
        if dataset.features:
            features = np.array(dataset.features)[inds]
        else:
            features = None
            
        return Dataset(dataset.X[:, inds], dataset.y, features, dataset.label)
    
    
    
    def fit_transform(self, dataset:Dataset) -> Dataset:
        """
        Calls the fit() and transform() methods, returning the filtered version
        of the given Dataset instance.
        
        Args
        ----------
        :param dataset (Dataset): Dataset object.
        """
        model = self.fit(dataset)
        return model.transform(dataset)
