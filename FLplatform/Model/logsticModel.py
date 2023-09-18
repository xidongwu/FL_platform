# imports
import numpy as np

class LogisticModel:
    """Logistic Regression.
    """

    def __init__(self, n_coords):
        self.n_coords = n_coords
        # self.para = np.random.rand(n_coords, 1)
        self.para = np.ones((n_coords, 1)) * 0.5

        # self.data = 0
        print('Logistic')

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def ComputeLoss(self, para, x, y):
        z = x @  para
        return np.mean(- z * y + np.log(1+ np.exp(z)))

    def ProximalOperator(self, para, gamma):
        def operator(n):
            sign = 1 if n > 0 else -1
            return  sign * max( abs(n) - gamma, 0)
        return np.expand_dims(np.fromiter( map(operator, para), float), axis=1)


    def NumParameters(self):
        return self.n_coords

    def PrecomputeCoefficients(self, para, x, y):
        return x.T @ (self.sigmoid(x @ para) - y)




class LogisticModelT:
    """Logistic Regression.

    """

    def __init__(self, n_coords):
        self.n_coords = n_coords
        self.para = np.random.rand(n_coords, 1)


    def ComputeLoss(self, para, x, y):

        obj = np.log(1 + np.exp(-y * (x @  para)))
        return np.mean(obj)


    def ProximalOperator(self, para, gamma):
        def operator(n):
            sign = 1 if n > 0 else -1
            return  sign * max( abs(n) - gamma, 0)
        return np.expand_dims(np.fromiter( map(operator, para), float), axis=1)


    def NumParameters(self):
        return self.n_coords

    def PrecomputeCoefficients(self, para, x, y):
        obj = np.log(1 + np.exp(-y * (x @  para)))
        return np.mean(obj)

    def logistic_grad(self, para, x, y):
        """
        para : dlx1
        xl: Nxdl
        y: Nx1
        grad: dlx1
        """
        e = np.exp(-y * (x @  para))
        coef = -y * e / (1 + e)
        grad = x.transpose() @ coef / len(coef)
        return grad