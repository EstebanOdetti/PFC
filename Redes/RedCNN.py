import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import scipy.io as sio
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
mat_fname = 'Datasets/mi_matriz.mat'
mat = sio.loadmat(mat_fname)
matriz_cargada = mat['dataset_matriz']
from torch.utils.data import TensorDataset, DataLoader


#primero mezclamos los casos
num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]
# Puedo hacer 138 casos de train y 58 de train (30%)
train = matriz_cargada_mezclada[0:137,:,:,1:16]
test = matriz_cargada_mezclada[138:,:,:,1:16]
temp_train = matriz_cargada_mezclada[0:137,:,:,17]
temp_test = matriz_cargada_mezclada[138:,:,:,17]

primeros_10_casos = temp_train[:10]
#intento de dibujo; NOSE SI ESTA BIEN
# for i in range(10):
#     caso = primeros_10_casos[i]
#     imagen = caso[:, :]  
#     plt.subplot(2, 5, i + 1)  
#     plt.imshow(imagen, cmap='gray')  
#     plt.axis('off')  
#     plt.title(f'Caso {i+1}')  
# plt.tight_layout()
# plt.show()  

class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        
        self.conv1 = nn.Conv2d(20, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 3 * 3, 1)  # Salida lineal para regresión
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
net=CNNRegressor()
# Tasa de aprendizaje inicial para el gradiente descendente
learning_rate = 0.0001
# Construimos el optimizador, y le indicamos que los parámetros a optimizar 
# son los del modelo definido: net.parameters()
optimmizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
# Definimos también la función de pérdida a utilizar
criterion = torch.nn.CrossEntropyLoss() 
# Creamos un loader iterable indicandole que debe leer los datos a partir de
# del dataset creado en el paso anterior. Este objeto puede ser iterado
loader = DataLoader(dataset=train, batch_size=3, shuffle=True)
# Número de épocas
num_epochs = 5
# Lista en la que iremos guardando el valor de la función de pérdida en cada 
# etapa de entrenamiento
loss_list = []


# Bucle de entrenamiento
for i in range(num_epochs):

    # Itero sobre todos los batches del dataset
    for x, y in loader:
        # Seteo en cero los gradientes de los parámetros a optimizar
        optimmizer.zero_grad()

        # Movemos los tensores a memoria de GPU, no hay GPU por ahora cpu
       # x = x.to(device)
       # y = y.to(device)
       
        # Realizo la pasada forward por la red
        loss = criterion(net(x), y)
        
        # Realizo la pasada backward por la red        
        loss.backward()
        
        # Actualizo los pesos de la red con el optimizador
        optimmizer.step()

        # Me guardo el valor actual de la función de pérdida para luego graficarlo
        loss_list.append(loss.data.item())

    # Muestro el valor de la función de pérdida cada 100 iteraciones        
    #if i > 0 and i % 100 == 0:
    print('Epoch %d, loss = %g' % (i, loss))