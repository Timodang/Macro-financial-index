from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract class to store the econometrics and machine learning
    models we will estimate as part of this project
    """

    @abstractmethod
    def model_estimate(self, y, x, window):
        """
        Method to estimate a model using a rolling window
        :param y: variable to estimate
        :param x: regressors
        :param window: window used for the computation
        :return:
        """
        pass
    @abstractmethod
    def predict_model(self, y, x):
        """
        method to predict the results from the model out-sample
        :param y:
        :param x:
        :return:
        """
        pass

    # Class for factor model

    # Class for ridge model

    # Class for group ridge model

    # Class for random forest model