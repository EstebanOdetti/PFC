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
from sklearn.metrics import mean_squared_error

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
    
device = torch.device('cuda:0')
net=CNN()
net = net.to(device)

learning_rate = 0.0001
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
criterion = torch.nn.MSELoss() 
num_epochs = 2000
loss_list = []

# Creamos un TensorDataset a partir de tus tensores
train_dataset = TensorDataset(train_tensor.permute(0, 3, 1, 2), temp_train_tensor) # Reordenamos las dimensiones aquí
test_dataset = TensorDataset(test_tensor.permute(0, 3, 1, 2), temp_test_tensor) # Reordenamos las dimensiones aquí

# Creamos DataLoaders a partir de los TensorDatasets
batch_size = 32  # Define el tamaño del batch que desees
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# Obtenemos el primer tensor del TensorDataset
first_tensor = train_loader.dataset.tensors[0]
# Imprimimos la forma del tensor
print(first_tensor.size())
# Bucle de entrenamiento
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):  # Ahora i es el índice de iteración
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        loss = criterion(net(x), y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # Imprimimos cada 100 iteraciones
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss = {loss}')


net.eval()  # Cambia el modo de la red a evaluación.

with torch.no_grad():  # No necesitamos calcular los gradientes.
    mse_list = []  # Lista para guardar los errores cuadráticos medios.

    for i in range(len(test_tensor)):
        inputs = test_tensor[i]
        real_output = temp_test_tensor[i]

        inputs = inputs.permute(2, 0, 1)  # Reordena las dimensiones a (canales, altura, ancho)
        inputs = inputs.to(device)
        real_output = real_output.to(device)


        # Pasamos los datos por la red (forward pass)
        predicted_output = net(inputs.unsqueeze(0)).squeeze()  # Asegúrate de eliminar la dimensión de batch con .squeeze()

        # Calculamos la pérdida de validación
        mse = mean_squared_error(real_output.cpu().detach().numpy(), predicted_output.cpu().detach().numpy())
        mse_list.append(mse)

    print(f'MSE on test data: {np.mean(mse_list)}')