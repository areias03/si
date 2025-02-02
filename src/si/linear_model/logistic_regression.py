import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistic.sigmoid_function import sigmoid_function
from sklearn import preprocessing
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    A linear regression model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique
    Parameters
    ----------
    :param l2_penalty: The L2 regularization parameter
    :param alpha: The learning rate
    :param max_iter: The maximum number of iterations
    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    cost_history: dictionary
        Cost value for every iteration.
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """
        Stores variables.
        Parameters
        ----------
        :param l2_penalty: The L2 regularization parameter
        :param alpha: The learning rate
        :param max_iter: The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def fit(self, dataset: Dataset, use_adaptive_fit: bool = False, scale:bool = True) -> 'LogisticRegression':
        """
        Fit the model to the dataset
        Parameters
        ----------
        :param dataset: An instance of the Dataset class to train the model.
        :param use_adaptive_fit: Boolean indicating whether the learning rate (alpha) should be altered
                                 as the cost value starts to stagnate.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        m, n = dataset.shape()
        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(int(self.max_iter)):
            # predicted y
            y_pred = sigmoid_function(np.dot(data, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, data)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # calculating cost in each iteration
            self.cost_history[i] = self.cost(dataset)

            # stopping criteria (version 1)
            if i > 0:
                if np.abs(self.cost_history[i] - self.cost_history[i - 1]) < 0.0001:
                    if use_adaptive_fit:
                        self.alpha /= 2
                    else:
                        break

        return self



    def predict(self, dataset: Dataset, scale:bool = True) -> np.array:
        """
        Predict the output of the dataset
        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        pred_vals = sigmoid_function(np.dot(data, self.theta) + self.theta_zero)
        mask = pred_vals >= 0.5
        pred_vals[mask] = 1
        pred_vals[~mask] = 0
        return pred_vals


    def score(self, dataset: Dataset, scale:bool = True) -> float:
        """
        Compute the Mean Square Error of the model on the dataset
        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        y_pred = self.predict(dataset, scale)
        return accuracy(dataset.y, y_pred)



    def cost(self, dataset: Dataset, scale:bool = True) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using regularization
        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        pred_vals = sigmoid_function(np.dot(data, self.theta) + self.theta_zero)
        m, n = dataset.shape()
        y = dataset.y
        regularization = self.l2_penalty/(2*m)*np.sum(self.theta**2)
        #for elem in y:
        #    if elem == 1:
        #        cond.append(np.log(y_pred))
        #    else:
        #        cond.append(np.log(1-y_pred))
        cond = np.log(1-pred_vals)

        return -1/m * np.sum(y*np.log(pred_vals) + (1-y)*cond) + regularization

    def plot_cost_history(self):
        """
        Plots the cost history as y axis and the number of iterations as the x axis.    
        """
        plt.plot(self.cost_history.keys(), self.cost_history.values(), "-k")
        plt.xlabel("Iteration number")
        plt.ylabel("Cost")
        plt.show()      
