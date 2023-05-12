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
from torch.utils.data import TensorDataset, DataLoader

print(os.getcwd())
mat_fname = 'Datasets/mi_matriz.mat'
mat = sio.loadmat(mat_fname)
matriz_cargada = mat['dataset_matriz']

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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 2 * 2, 49)  # Cambiar 64 a 49 (7 * 7)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        x = x.view(x.size(0), 7, 7)  # Añadir esta línea para cambiar la forma de x a (batch_size, 7, 7)
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
criterion = torch.nn.MSELoss() 
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
num_epochs = 1000
# Lista en la que iremos guardando el valor de la función de pérdida en cada 
# etapa de entrenamiento
loss_list = []
# Bucle de entrenamiento
for i in range(num_epochs):
    for j in range(136):
        optimmizer.zero_grad()
        x = train_tensor[j]
        y = temp_train_tensor[j]

        x = x.permute(2, 0, 1) # Reorder dimensions to (channels, height, width)

        x = x.to(device)
        y = y.to(device)
        #print(x.shape)
        #print(y.shape)
       
        # Realizo la pasada forward por la red
        loss = criterion(net(x.unsqueeze(0)), y) # Add batch dimension to x
        # Realizo la pasada backward por la red        
        loss.backward()
        
        # Actualizo los pesos de la red con el optimizador
        optimmizer.step()

        # Me guardo el valor actual de la función de pérdida para luego graficarlo
        loss_list.append(loss.data.item())
        # Muestro el valor de la función de pérdida
        print('Epoch %d, loss = %g' % (i, loss))
    