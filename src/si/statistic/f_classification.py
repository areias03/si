import numpy as np
from scipy import stats

from si.data.dataset import Dataset

def f_classification(dataset: Dataset):
    """
    Scoring function for classification problems that computes one-way ANOVA F-value for the
    provided dataset.

    Args:
        dataset (_type_): Dataset object.

    Returns:
        F (np.array): F scores.
        p (np.array): p-values.
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes] 
    F, p = stats.f_oneway(*groups)
    return F, p
