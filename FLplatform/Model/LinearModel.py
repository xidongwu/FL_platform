# imports
import numpy as np
import gflags


Flags = gflags.FLAGS


class LinearModel:
    """Linear Regression.

    Parameters
    ----------
    n_coords : int, feature dimension 
    n_iterations : int
        No of passes over the training set

    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration

    """

    def __init__(self, n_coords):
        self.n_coords = n_coords
        # self.para = np.random.rand(n_coords, 1)
        self.para = np.zeros((n_coords, 1))

        self.data = 0

        print("linear model")


    def ComputeLoss(self, para, x , y): 
        # print(np.shape(x))
        # print(np.shape(para))

        y_pre = x @  para - y
        return (y_pre.T @ y_pre)[0][0] / np.shape(x)[0] + np.linalg.norm( para, 1) * Flags.l1_lambda

    def ProximalOperator(self, para, gamma):
        def operator(n):
            sign = 1 if n > 0 else -1
            return  sign * max( abs(n) - gamma, 0)
        return np.expand_dims(np.fromiter( map(operator, para), float), axis=1)


    def NumParameters(self):
        return self.n_coords

    # def PrecomputeCoefficients(self, model, x, y, left_right):
    #     tmp = x[range(left_right[0],left_right[1]), :] @ self.para
    #     tmp2 = (tmp - y[range(left_right[0],left_right[1]), :]) / (left_right[1] - left_right[0])
    #     return x[range(left_right[0],left_right[1]), :].T @ tmp2
    def PrecomputeCoefficients(self, para, x, y):

        return x.T @ (x @  para - y)
