import numpy as np

def sigmoid_function(X:np.ndarray) -> float:
    """
    Implements the sigmoid function to an array of values
    
    Parameters
    ----------
    :param X (_type_): Array of values

    Returns
    -------
    float
        Probability of values being equal to 1 (sigmoid function)
    """

    return 1/(1+(np.exp(-X)))
