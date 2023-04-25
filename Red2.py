import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

datos_totales = pd.read_csv(r'C:\Users\Usuario\Desktop\PyThorch Test\datos_sinteticos_ver_3.csv')
# Seleccionamos las columnas deseadas
columnas_deseadas = ["x", "y", "borde", "model.k", "model.G", "model.c", "BN", "BS", "Be", "Bo", "columna_DIR1", "columna_DIR2", "columna_NEU1", "columna_NEU2", "columna_ROB1", "columna_ROB2"]
datos = datos_totales[columnas_deseadas]

# Seleccionamos la columna PHI_temp
datos_PHI_temp = datos_totales["PHI_temp"]

# Seleccionamos las columnas Q_tempx y Q_tempy
datos_Q_tempx = datos_totales["Q_tempx"]
datos_Q_tempy = datos_totales["Q_tempy"]

# Usamos train test split para hacer esquema 80-20
datos_tr, datos_ts, salida_esperada_tr, salida_esperada_ts = train_test_split(datos, datos_PHI_temp, test_size=0.2,
                                                                              random_state=0)
#primero a numpy
datos_tr = datos_tr.to_numpy()
datos_ts =datos_ts.to_numpy()
salida_esperada_tr =salida_esperada_tr.to_numpy()
salida_esperada_ts =salida_esperada_ts.to_numpy()
# Dispositivo en que se ejecturá el modelo: 'cuda:0' para GPU y 'cpu' para CPU
device = torch.device('cuda:0')

batch_size = 64
net = Net()
net = net.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Entrenamiento de la red neuronal
for epoch in range(1000):

    running_loss = 0.0
    for i in range(datos_tr.shape[0] // batch_size):
        # Obtener un lote de datos
        batch_X = datos_tr[i*batch_size:(i+1)*batch_size]
        batch_y = salida_esperada_tr[i*batch_size:(i+1)*batch_size]

        # Resetear los gradientes
        optimizer.zero_grad()

        # Hacer una predicción
        outputs = net(batch_X)

        # Calcular la pérdida
        loss = criterion(outputs.view(-1), batch_y)

        # Retropropagar el error y actualizar los pesos
        loss.backward()
        optimizer.step()

        # Acumular la pérdida
        running_loss += loss.item()

    # Calcular la pérdida promedio para esta época
    epoch_loss = running_loss / (X_train.shape[0] // batch_size)

    # Imprimir la pérdida promedio para esta época
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))



