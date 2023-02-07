import numpy as np
from numpy.linalg import inv
from abc import ABC, abstractmethod

class MlM(ABC):
    @abstractmethod
    def __init__(self, loss):
        pass

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class Loss(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def loss(self, X, Y, W):
        pass

    @abstractmethod
    def d_loss(self, X, Y, W):
        pass

class SinglePassLoss(Loss, ABC):
    @abstractmethod
    def finalJ(self, X, Y, W):
        pass