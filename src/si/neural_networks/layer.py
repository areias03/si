import numpy as np
from si.statistic.sigmoid_function import sigmoid_function

class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.
    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        return error


class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        self.X = X
        return 1 / (1 + np.exp(-X))

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        sigmoid_deriv = sigmoid_function(self.X) * (1 - sigmoid_function(self.X))
        
        error_to_propagate = error * sigmoid_deriv
        return error_to_propagate 


class ReLUActivation:
    """
    A class to create a ReLU activation function to compile onto a
    Neural Network class ('NN') instance.
    """
    def __init__(self):
        self.X = None

    def forward(self, input_data: np.array):
        """
        Applies the ReLU activation function to the input data
        (returns the maximum value between the input data and 0).
        Parameters
        ----------
        :param input_data: The resulting values of a neural network layer (or input data)
        """
        self.X = input_data
        return np.maximum(input_data, 0)


    def backward(self, error:np.ndarray, learning_rate:bool = 0.001):
        """
        Returns the input error value
        Parameters
        ----------
        :param error: The value of the error function derivative
        :param learning_rate: A boolean value to influence the weight values update
                              (smaller values update the weights at a slower rate)
        """
        error_to_propagate = np.where(self.X > 0, 1, 0)
        return error_to_propagate
    

class SoftMaxActivation:
    def __init__(self):
        pass
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Computes the probability of each class.
        Args:
            X (np.ndarray): Input data
        Returns:
            np.ndarray: The probability of each class
        """
        
        ziexp = np.exp( - np.max(X))
        return (ziexp / (np.sum(ziexp, axis=1, keepdims=True)))


class LinearActivation:
    def __init__(self) -> None:
        pass
    
    def forward(X: np.ndarray) -> np.ndarray:
        """
        Computes the linear activation, also known as "No activation".
        :param X: input data
        :return: Returns the input data. The linear activation basically spits out the input data as it is.
        """
        
        return X
