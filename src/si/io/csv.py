import sys
sys.path.append('.')
sys.path.append("/Users/alexandre/Documents/Mestrado/2º\ Ano/SIB/si/src/si/data")


from typing import Optional
import pandas as pd
import numpy as np

from data.dataset import Dataset

def read_csv(filename,sep,features,labels):
    """
    Reads a csv file and returns a Dataset object of that file.
    Args:
        filename (str): File path
        sep (str): Separator between values. Defaults to "," .
        features (Optional[bool], optional): If the csv file has feature names. Defaults to True.
        label (int): If the dataset has defined labels. Defaults to False.
    Returns:
        Dataset: The dataset object
    """
    data = pd.read_csv(filename, sep)
    if features and label: 
        y = data.iloc[:, -1] 
        label = data.columns[-1]
        data = data.iloc[:, :-1]
        features = data.columns

    elif features and label is False:  
        features = data.columns
        y = None

    elif features is False and label:  
        y = data.iloc[:, -1]
        label = data.columns[-1]
        data = data.iloc[:, :-1]

    else: 
        y = None

    return Dataset(data, y, features, label)


def write_csv(filename, dataset,sep,features,label):
    """
    Writes a csv file from a dataset object.
    Args:
        dataset (_type_): Dataset to save on csv format
        filename (str): Name of the csv file that will be saved
        sep (str, optional): Separator of values. Defaults to ",".
        features (Optional[bool], optional): If the dataset object has feature names. Defaults to True.
        label (Optional[bool], optional): If the dataset object has label names Defaults to True.
    """
    csv = pd.DataFrame(data=dataset.X)
    
    if features:
        csv.columns = dataset.features
    
    if label:
        csv.insert(loc=0, column=dataset.label, value=dataset.y)
        
    csv.to_csv(filename, sep = sep, index=False)
