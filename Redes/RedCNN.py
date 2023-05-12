import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

datos_totales = pd.read_csv(r'C:\Users\Usuario\Desktop\Proyectos\PyTorch\PyThorch Test\Datasets\datos_sinteticos_matriz_ver_6.csv')


print(datos_totales.shape)

class NetCNN(nn.Module):

    def __init__(self):
        super(NetCNN, self).__init__()
        # las imagenes son de 28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Al pasar de capa convolucional a capa totalmente conectada, tenemos
        # que reformatear la salida para que se transforme en un vector unidimensional
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        