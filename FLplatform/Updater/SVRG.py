# imports
import numpy as np
import gflags
from mpi4py import MPI

comm = MPI.COMM_WORLD
Flags = gflags.FLAGS

class SVRG:

    def __init__(self, coord, x, y):
        self.coord = coord
        self.copy_para = np.random.rand(coord, 1)
        self.copy_grad = np.random.rand(coord, 1)
        self.x = x
        self.y = y
        print('here is SVRG')

    def EpochBegin(self, model):
    #     self.modelcopy(model, x, y)
    # def modelcopy(self, model, x, y):
        self.copy_para = model.para

        # local_grad = model.PrecomputeCoefficients(model.para, x, y)
        if comm.Get_rank() != 1:
            local_grad = self.x.T @ (1 / (1 + np.exp(-self.x @ model.para)) - self.y)
            tmp_grad = np.zeros((self.coord + 1, 1))
            tmp_grad[:-1, :] = local_grad
            tmp_grad[-1, 0] = np.shape(self.x)[0]

            # print("shape", np.shape(local_grad))
            send_req = comm.Isend([tmp_grad, MPI.DOUBLE], dest=1, tag=301)
            send_req.wait()

            recv_req = comm.Irecv([self.copy_grad, MPI.DOUBLE], source = 1, tag= 401)
            recv_req.wait()
            # print(" wancheng ", comm.Get_rank())
        else:
            grad1 = np.zeros((self.coord + 1, 1))
            grad2 = np.zeros((self.coord + 1, 1))

            recv_req1 = comm.Irecv([grad1, MPI.DOUBLE], source = 2, tag= 301)
            recv_req1.wait()
            recv_req2 = comm.Irecv([grad2, MPI.DOUBLE], source = 3, tag= 301)
            recv_req2.wait()
            self.copy_grad = (grad1[:-1, :] + grad2[:-1, :]) / (grad1[-1, 0] + grad2[-1, 0])

            send_req1 = comm.Isend([self.copy_grad, MPI.DOUBLE], dest=2, tag=401)
            send_req1.wait()

            send_req2 = comm.Isend([self.copy_grad, MPI.DOUBLE], dest=3, tag=401)
            send_req2.wait()
        # comm.Bcast([self.copy_grad, MPI.DOUBLE], root=1)
                # print("update done")

    def Update(self, model, x, y):

        # print("x shape", np.shape(x))
        # print("y shape", np.shape(y))
        # print("para shape", np.shape(model.para))
        grad_1 = x.T @ (1 / (1 + np.exp(x @ model.para)) - y)
        # grad_1 = model.PrecomputeCoefficients(model.para, x, y)
        grad_2 = x.T @ (1 / (1 + np.exp(x @ self.copy_para)) - y)
        # grad_2 = model.PrecomputeCoefficients(self.copy_para, x, y)
        grad = (grad_1 - grad_2) / Flags.mini_batch + self.copy_grad
        return grad
        # grad = grad_1 / Flags.mini_batch

        # grad /= Flags.mini_batch
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



















