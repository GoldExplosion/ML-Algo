from classes import *

class LinearRegression(MlM):
    '''
    Linear Regression
    '''
    def __init__(self, loss="LMS"):
        lossFunc = {
            "LMS" : LMS
        }
        self.J = lossFunc[loss]()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n = X[0].shape[1]
        m = X.shape[0]
        w = np.random.rand(n, 1)
        self.W = np.array(w*m)
        self.J.finalJ(X, Y, self.W)

    def predict(self, X):
        return np.dot(X,self.W)

class LMS(SinglePassLoss):
    '''
    Least Mean Squared Loss
    '''
    def __init__(self):
        pass

    def loss(self, X, y, W):
        inter = np.dot(X, W) - y
        return np.dot(inter, inter)/2

    def d_loss(self):
        pass

    def finalJ(self, X, y, W):
        return np.dot(inv(np.dot(X, X.T)), np.dot(X, y))

