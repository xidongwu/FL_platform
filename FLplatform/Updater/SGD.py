# imports
import numpy as np
import gflags

Flags = gflags.FLAGS

class SGD:

    def __init__(self, coord, x, y):
        self.x = x
        self.y = y
        # pass
        print('here is SGD')
    
    def EpochBegin(self, model):
        pass


    def Update(self, model, x, y):

        grad = x.T @ (1 / (1 + np.exp(-x @ model.para)) - y)
        # grad = model.PrecomputeCoefficients(model.para, x, y)
        grad /= Flags.mini_batch

        return grad
        # print('1111')
        # print(grad)

        # print('2222')
        # print(model.para)
        # print(Flags.learning_rate)
        
        # model.para -= Flags.learning_rate  * grad
        # # print('3333')
        # # print(model.para)
        # model.para = model.ProximalOperator(model.para, Flags.learning_rate * Flags.l1_lambda)
        # return model.para


















