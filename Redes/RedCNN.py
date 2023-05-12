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
import os
print(os.getcwd())
mat_fname = 'Datasets/mi_matriz.mat'
mat = sio.loadmat(mat_fname)
matriz_cargada = mat['dataset_matriz']
from torch.utils.data import Dataset, DataLoader


#primero mezclamos los casos
num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]
# Puedo hacer 138 casos de train y 58 de train (30%)
train = matriz_cargada_mezclada[0:137,:,:,0:16]
test = matriz_cargada_mezclada[138:,:,:,0:16]
temp_train = matriz_cargada_mezclada[0:137,:,:,17]
temp_test = matriz_cargada_mezclada[138:,:,:,17]
print(train.shape)
train_tensor = torch.from_numpy(train).float()
test_tensor = torch.from_numpy(test).float()
temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()

train_tensor_reshape = train_tensor.permute(2, 0, 1).contiguous()

#primeros_10_casos = temp_train[:10]
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
        
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * height * width, 1)  # Ajustar el tamaño de entrada para self.fc
        
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  
        # Capas convolucionales
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        # Capa de agrupación (max pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Capa completamente conectada
        self.fc = nn.Linear(32 * 2 * 2, 64)
        
    def forward(self, x):
        # Paso a través de la capa convolucional
        x = self.conv1(x)
        # Aplicar función de activación
        x = torch.relu(x)
        # Aplicar capa de agrupación
        x = self.pool(x)
        # Aplanar la salida de la capa de agrupación
        x = x.view(x.size(0), -1)  # Corrección aquí
        # Paso a través de la capa completamente conectada
        x = self.fc(x)
        return x

# Dispositivo en que se ejecturá el modelo: 'cuda:0' para GPU y 'cpu' para CPU
#device = torch.device('cpu')
device = torch.device('cuda:0')
net=CNN()
net = net.to(device)
# Tasa de aprendizaje inicial para el gradiente descendente
learning_rate = 0.0001
# Construimos el optimizador, y le indicamos que los parámetros a optimizar 
# son los del modelo definido: net.parameters()
optimmizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
# Definimos también la función de pérdida a utilizar
criterion = torch.nn.CrossEntropyLoss() 
# Creamos un loader iterable indicandole que debe leer los datos a partir de
# del dataset creado en el paso anterior. Este objeto puede ser iterado
##Pero para nuestro caso tenemos que definir el dataset
#class CustomDataset(Dataset):
 #   def __init__(self, data):
  #      self.data = data
        
   # def __getitem__(self, index):
    #    x = torch.from_numpy(self.data[index, :,:,0:16])
     #   y = torch.from_numpy(self.data[index, :,:,17])
      #  return x.to(device), y.to(device) 
   
    #def __len__(self):
     #   return self.data.shape

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loader = CustomDataset(train)
#loader = DataLoader(dataset=train, batch_size=6, shuffle=True)
#print(len(loader))
#print(len(loader.dataset))
# Número de épocas
num_epochs = 5
# Lista en la que iremos guardando el valor de la función de pérdida en cada 
# etapa de entrenamiento
loss_list = []
# Bucle de entrenamiento
for i in range(num_epochs):
    #print(loader.data[i,1,1,0])
    # Itero sobre todos los batches del dataset
    
    for j in range(136):
        # Seteo en cero los gradientes de los parámetros a optimizar
        
        optimmizer.zero_grad()
        x = train_tensor[j]
        y = temp_train_tensor[j]
        # Movemos los tensores a memoria de GPU o CPU
        x = x.to(device)
        y = y.to(device)
        ##quiero ver la forma de cada elemento
        print(x.shape)
        print(y.shape)
       
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
    #print('Epoch %d, loss = %g' % (i, loss))
    