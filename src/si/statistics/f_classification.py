from si.data.dataset import Dataset

def f_classification(dataset: Dataset):

    classes = Dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes] 
    F, p = stats.f_oneway(*groups)
    return F, p
