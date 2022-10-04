from si.data.dataset import Dataset

class VarianceThreshold:

    def __init__(self,threshold):
        self.threshold = threshold
        self.variance = variance

    def fit(self,dataset):
        variance = dataset.get_variance()
        self.variance = variance
        return self

    def transform(self,dataset: Dataset):

        X = dataset.X

        features_mask = self.variance > self.threshold
