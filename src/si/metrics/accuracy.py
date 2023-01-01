import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the error between arguments given using the accuracy formula:
    
                    (VN + VP) / (VN + VP + FP + FN)
    Args:
        y_true (np.ndarray): real values.
        y_pred (np.ndarray): predicted values.
    Returns:
        float: Error value between y_true and y_pred
    """
    
    return np.sum(y_true == y_pred) / len(y_true)
