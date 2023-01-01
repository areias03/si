import numpy as np
from cmath import sqrt

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Function that calculates the Root Mean Squared Error (RMSE) metric.
    Args:
        y_true (np.ndarray): Real values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: RMSE between real and predicted values.
    """

    rmse = sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

    return rmse 
