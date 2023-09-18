# mpirun -n 4 python main.py  

import matplotlib
matplotlib.use('tkagg')

import numpy as np
import gflags
from mpi4py import MPI
import sys

import pandas as pd

from Model.LinearModel import *
from Model.logsticModel import *
from Model.DNNModel import *

import time
from Updater.SGD import *
from Updater.SVRG import *

from Trainer.ServerTrainer import *
from Trainer.WorkerTrainer import *
from Trainer.ServerTrainerDNN import *
from Trainer.WorkerTrainerDNN import *

from tkinter import *
from tkinter import messagebox
# from mpl_toolkits.mplot3d import Axes3D

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


Flags = gflags.FLAGS
#data
gflags.DEFINE_string('data_file', 'blank', 'Input data file')
# model
gflags.DEFINE_boolean('least_l2_l1', False, 'least square loss with l2 and l1 norm regularization type')
gflags.DEFINE_boolean('logistic_l2_l1', False, 'logistic loss with l2 and l1 norm regularization type')
gflags.DEFINE_boolean('DNN', False, 'DNN with l2 and l1 norm regularization type')

gflags.DEFINE_boolean('svrg', False, 'Use SVRG')
gflags.DEFINE_boolean('sgd', False, 'least square loss with l2 and l1 norm regularization type')

#parameter
gflags.DEFINE_float('yaxis', 20, 'Number of passes of data in training')

gflags.DEFINE_integer('n_epochs', 100, 'Number of passes of data in training')
gflags.DEFINE_integer('mini_batch', 1, 'mini batch size in each epoch')
gflags.DEFINE_integer('in_iters', 100, 'Inside iterations')
gflags.DEFINE_integer('max_delay', 100, 'max delay for each worker')
gflags.DEFINE_integer('num_workers', 2, 'Number of workers in in the cluster')
gflags.DEFINE_integer('group_size', 1, 'group size of workers received by server in each inner iteration')
gflags.DEFINE_integer('l1_lambda', 0, 'regularization parameter for l1 norm')
gflags.DEFINE_integer('dim1', 618, 'regularization parameter for l2 norm')

gflags.DEFINE_float('learning_rate', .0001, 'Learning rate')

def data_loader(path, header):
    marks_df = pd.read_csv(path, header = header)
    return marks_df


def generate_data_set(taskID):
 
    np.random.seed(taskID)
    x = np.random.rand(100, 2)
    x = np.concatenate((x, np.ones((100,1), dtype=int)), axis=1)
    y = 2 * x[:, 0:1]+ 3 * x[:, 1:2] + 4 * x[:, 2:3] + np.random.rand(100, 1)
    return x, y


if __name__ == "__main__":
    # main(sys.argv)
    Flags(sys.argv)

    # Initialization 
    comm = MPI.COMM_WORLD
    taskID = comm.Get_rank()
    nprocs = comm.Get_size()

    if taskID == 0:
        import matplotlib.pyplot as plt
        # # 创建画布需要的库
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = Tk()
        root.title("Secured Federated Learning Client")
        root.geometry('1200x900')

        ## data stats #########
        Label(root, text='Data Information', font=('Helvetica',20,'bold','underline')).place(x=200, y=50, width=300, height=50)

        clicked1 = StringVar()
        clicked1.set("Select Data")
        data = OptionMenu(root, clicked1, "bisolet", "others")
        data.place(x=250, y=100, anchor='nw')
        # datatext = StringVar()
        # datatext.set("Data Description")

        DataLabel1 = Label(root, bg="#e4e6e8", text="Data Description",  anchor='nw').place(x=50, y=130, width=500, height=200)
        def datainfo(info):
            if info == "bisolet":
                return "--bisolet:  \n" + "Info: Data used for test"

        def datashow():
            DataLabel = Label(root, bg="#e4e6e8", text=datainfo(clicked1.get()),  anchor='nw').place(x=50, y=130, width=500, height=200)

        dataButton = Button(root, text="Apply", pady=50, command=datashow).place(x=450, y=100, width=50, height=20)

        ## sys stats #########
        Label(root, text='System Configuration', font=('Helvetica',20,'bold','underline')).place(x=800, y=50, width=300, height=50)

        # Label(root, text='#chat', font=('Helvetica',18)).place(x=750, y=100, width=50, height=30)
        # Label(root, text='#CPU', font=('Helvetica',18)).place(x=1000, y=100, width=50, height=30)
        # Label(root, text='#GPU', font=('Helvetica',18)).place(x=750, y=150, width=50, height=30)
        # Label(root, text='#Memory', font=('Helvetica',18)).place(x=1000, y=150, width=80, height=30)

        var1 = IntVar()
        var2 = IntVar()
        var3 = IntVar()
        var4 = IntVar()

        sys1 = Checkbutton(root, text='CPU', variable=var1, font=('Helvetica',18)).place(x=750, y=100, width=100, height=30)
        sys2 = Checkbutton(root, text='GPU', variable=var2, font=('Helvetica',18)).place(x=1000, y=100, width=100, height=30)
        # sys3 = Checkbutton(root, text='#GPU', variable=var3, font=('Helvetica',18)).place(x=750, y=150, width=100, height=30)
        # sys4 = Checkbutton(root, text='#Memory', variable=var4, font=('Helvetica',18)).place(x=1000, y=150, width=160, height=30)

        Label(root, text='#Default path: Desktop/FederatedLearning/', font=('Helvetica',18)).place(x=750, y=150, width=360, height=30)

        Label(root, text='#Save as', font=('Helvetica',18)).place(x=750, y=200, width=90, height=30)
        e0 = Entry(root, width=25)
        e0.place(x=850, y=200)

        # e0.grid(row=0, column=0, columnspan=2, rowspan=3, padx=750,pady=300)

        Label(root, text='Model Configuration', font=('Helvetica',20,'bold','underline')).place(x=800, y=300, width=300, height=50)
        
        Label(root, text='Model', font=('Helvetica',16)).place(x=760, y=350, width=50, height=30,  anchor='nw')
        clicked2 = StringVar()
        clicked2.set("Model")
        Model2 = OptionMenu(root, clicked2, "Least Square", "Logistic Regression", "DNN")
        Model2.place(x=750, y=380, width=150, anchor='nw')

        Label(root, text='Optimizer', font=('Helvetica',16)).place(x=1010, y=350, width=70, height=30)
        clicked3 = StringVar()
        clicked3.set("Optimizer")
        Model3 = OptionMenu(root, clicked3, "SGD", "SVRG")
        Model3.place(x=1000, y=380, width=150, anchor='nw')

        Label(root, text='#Clients', font=('Helvetica',16)).place(x=760, y=430, width=70, height=30)
        clicked4 = StringVar()
        clicked4.set("Clients")
        Model3 = OptionMenu(root, clicked4, "2", "3", "4", "5", "6")
        Model3.place(x=750, y=460, width=150, anchor='nw')


        Label(root, text='Training Configuration', font=('Helvetica',20,'bold','underline')).place(x=800, y=550, width=300, height=50)

        Label(root, text='#Epochs', font=('Helvetica',18)).place(x=750, y=600, width=100, height=30)
        e1 = Entry(root, width=10)
        e1.grid(row=1, column=0, columnspan=2, rowspan=3, padx=750,pady=630)

        Label(root, text='#Batch Size', font=('Helvetica',18)).place(x=1000, y=600, width=100, height=30)
        e2 = Entry(root, width=10)
        e2.grid(row=1, column=1, columnspan=2, rowspan=3, padx=1000,pady=630)

        Label(root, text='#Inter Loop', font=('Helvetica',18)).place(x=750, y=700, width=100, height=30)
        e3 = Entry(root, width=10)
        e3.grid(row=2,column=0, columnspan=2, rowspan=3, padx=750,pady=730) 

        Label(root, text='#Step Size', font=('Helvetica',18)).place(x=1000, y=700, width=100, height=30)
        e4 = Entry(root, width=10)
        e4.grid(row=2,column=1, columnspan=2, rowspan=3, padx=1000,pady=730) 

        # e5 = Entry(root, width=10)
        # e5.grid(row=3,column=0, columnspan=2, rowspan=3, padx=750,pady=230) 


        # 创建一个容器, 没有画布时的背景
        frame1 = Frame(root, bg="#ffffff")
        frame1.place(x=50, y=400, width=600, height=450)
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        fig = plt.figure(figsize=(6, 4.5), edgecolor='blue')

        ax = fig.add_subplot(1,1,1)

        # 定义刻度
        # ax.set_xlim(0, 100)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, Flags.yaxis)
        plt.xlabel("Time (s)")
        plt.ylabel("Loss")

        # ax.set_zlim(0, 100)
        canvas = FigureCanvasTkAgg(fig, master=frame1)
        canvas.draw()
        # 显示画布
        canvas.get_tk_widget().place(x=0, y=0)

        i = 0
        # 定义存储坐标的空数组
        x = []
        y = []

        # 抛物线动态绘制函数
        def drawImg():

            data = np.array([0.0, 0.0])
            comm.Bcast(data, root=0)
            data = comm.recv(source=1, tag=77)
            print("drawing")

            xi = data[0]
            yi = data[1]
            global ax
            ax.clear()
            ax.set_xlim(0, 50)
            ax.set_ylim(0, Flags.yaxis)
            plt.xlabel("Time (s)")
            plt.ylabel("Loss")
            # ax.set_ylim(0, 20)

            # ax.autoscale_view()

        #     # ax.set_zlim(0, 100)
            global x
            global y
        #     global z
            # t = i * 0.1
            # dtax = 20 * t * np.sin(0.25 * np.pi)
            x.append(xi)
            # dtay = t**2
            y.append(yi)
            ax.plot(x, y)
            canvas.draw()
            global afterHandler
            afterHandler = root.after(1, drawImg)

        def terminate():
            global afterHandler
            if afterHandler:
                root.after_cancel(afterHandler)
                afterHandler = None

        def stop():
            global x
            global y
            global ax
            ax.clear()
            x = []
            y = []
            data = np.array([1, 0.0])
            # comm.ssend(data, dest=1, tag=88)

            global afterHandler
            if afterHandler:
                root.after_cancel(afterHandler)
                afterHandler = None
            data = np.array([1, 0.0])
            comm.Bcast(data, root=0)

        def setup():
            # print("weishenme")
            print("data", clicked1.get())
            exe = np.zeros(6)

            if clicked2.get() == "Least Square":
                exe[0] = 0
                Flags.yaxis = 20
                # Flags.least_l2_l1 = True
            elif  clicked2.get() == "Logistic Regression":
                exe[0] = 1
                Flags.yaxis = 20
            else:
                exe[0] = 2
                Flags.yaxis = 0.7

                # Flags.logistic_l2_l1 = True
                # print(clicked2.get() )
                # print(Flags.logistic_l2_l1)

            if clicked3.get() == "SVRG":
                exe[1] = 0
                # Flags.svrg = True
                print("exe 0", clicked3.get())
            else:
                exe[1] = 1
                # Flags.sgd = True
                print("exe 1", clicked3.get())

            exe[2] = int(e1.get())
            # Flags.n_epochs = int(e1.get())
            exe[3] = int(e2.get())
            # Flags.mini_batch = int(e2.get())
            exe[4] = int(e3.get())
            # Flags.in_iters = int(e3.get())
            exe[5] = float(e4.get())
            # Flags.learning_rate = float(e4.get())

            comm.Bcast(exe, root=0)

            # print("e1", Flags.n_epochs)
            # print("e2", Flags.mini_batch)
            # print("e3", Flags.in_iters)
            # print("e4", Flags.learning_rate)

        setButton = Button(root, text="Apply", pady=50, command=setup).place(x=700, y=800, width=450, height=50)
        runbutton = Button(root, text="Run", pady=50, command=drawImg).place(x=50, y=350, width=180, height=50)
        terminate = Button(root, text="Pause", pady=50, command=terminate).place(x=260, y=350, width=180, height=50)
        stopbutton = Button(root, text="Stop", pady=50, command=stop).place(x=470, y=350, width=180, height=50)

        root.mainloop()

    else:
        while True: 
            exe = np.zeros(6)
            comm.Bcast(exe, root=0)
            print(exe)

            if exe[0] == 0:
                Flags.least_l2_l1 = True
                Flags.logistic_l2_l1 = False
                Flags.DNN = False
                Flags.yaxis = 20
            elif exe[0] == 1:
                Flags.logistic_l2_l1 = True
                Flags.least_l2_l1 = False
                Flags.DNN = False             

            else:
                Flags.logistic_l2_l1 = False
                Flags.least_l2_l1 = False
                Flags.DNN = True
                Flags.yaxis = 0.65
            
            if exe[1] == 0:
                Flags.svrg = True
                Flags.sgd = False
            else:
                Flags.sgd = True
                Flags.svrg = False

            Flags.n_epochs = int(exe[2])
            Flags.mini_batch = int(exe[3])
            Flags.in_iters = int(exe[4]) 
            Flags.learning_rate = exe[5] 

            data = pd.read_csv("data/bisolet.txt", header = None)

            length = len(data) / (nprocs - 2)

            # Flags.dim1 = 3 # !!!!!
            Flags.dim1 = 618 # !!!!!
            if Flags.DNN:
                Flags.dim1 = 617


            if taskID != 1:
            # X = feature values, all the columns except the last column
                X = data.iloc[int((taskID - 2) * length) : int((taskID - 1) * length), :-1].to_numpy()
                # y = target values, last column
                y = data.iloc[int((taskID - 2) * length) : int((taskID - 1) * length), -1].to_numpy()
                if Flags.DNN == False:
                    X = np.c_[X, np.ones((X.shape[0], 1))]
                    y = np.expand_dims(y, axis = 1)
            else:
                X = np.zeros((Flags.dim1, 1))
                y = np.zeros((Flags.dim1, 1))

            if Flags.DNN:
                print("herer")
                model = DNNModel(Flags.dim1, 32, 2)
                model.load_state_dict(torch.load("state_dict_model.pt"))


                if Flags.svrg:
                    model.opt = 0
                else:
                    model.opt = 1 
                if taskID == 1:
                    trainer = ServerTrainerDNN(model, X, y)
                    # print('Hello', taskID)
                else:
                    trainer = WorkerTrainerDNN(model, X, y)
                trainer.train()

            else: 
                if Flags.least_l2_l1: 
                    model = LinearModel(Flags.dim1)
                elif Flags.logistic_l2_l1:
                    print("should be here")
                    model = LogisticModel(Flags.dim1)

                if Flags.svrg:
                    updater = SVRG(Flags.dim1, X, y)
                elif Flags.sgd:
                    updater = SGD(Flags.dim1, X, y)
                else:
                    updater = SVRG(Flags.dim1, X, y)

                if taskID == 1:
                    trainer = ServerTrainer(model, X, y)
                    # print('Hello', taskID)
                else:
                    trainer = WorkerTrainer(model, X, y)
                trainer.train(updater)

