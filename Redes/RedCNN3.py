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
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader


mat_fname = 'Datasets/mi_matriz_solo_diritletch'
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
train_tensor = torch.from_numpy(train).float()
test_tensor = torch.from_numpy(test).float()
temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()

primeros_10_casos = temp_train_tensor[34:44]
for i in range(10):
    caso = primeros_10_casos[i]
    imagen = caso[:, :]  
    plt.subplot(2, 5, i + 1)  
    plt.imshow(imagen, cmap='gray')  
    plt.axis('off')  
    plt.title(f'Caso {i+1}')  
plt.tight_layout()
plt.show()  

# Preparar los datos de entrada
input_data = np.zeros_like(train)
input_data[:, :, 0, :] = train[:, :, 0, :]
input_data[:, :, -1, :] = train[:, :, -1, :]
input_data[:, 0, :, :] = train[:, 0, :, :]
input_data[:, -1, :, :] = train[:, -1, :, :]

input_tensor = torch.from_numpy(input_data).float().permute(0, 3, 1, 2)
# Seleccionar el primer canal para graficarlo
primer_canal = input_data[:, :, :, 0]
# Graficar los primeros 10 casos del primer canal de los datos de entrada después de aplicar las condiciones de borde
primeros_10_casos = primer_canal[34:44]
for i in range(10):
    caso = primeros_10_casos[i]
    imagen = caso  # No se necesita [:, :]
    plt.subplot(2, 5, i + 1)
    plt.imshow(imagen, cmap='gray')
    plt.axis('off')
    plt.title(f'Caso {i+1}')
plt.tight_layout()
plt.show()
# Preparar los datos de prueba
test_data = np.zeros_like(test)
test_data[:, :, 0, :] = test[:, :, 0, :]
test_data[:, :, -1, :] = test[:, :, -1, :]
test_data[:, 0, :, :] = test[:, 0, :, :]
test_data[:, -1, :, :] = test[:, -1, :, :]
test_tensor = torch.from_numpy(test_data).float().permute(0, 3, 1, 2)

# Crear TensorDatasets y DataLoaders
train_dataset = TensorDataset(input_tensor, temp_train_tensor)
test_dataset = TensorDataset(test_tensor, temp_test_tensor)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
def custom_loss(y_pred, y_true, input_data):
    # Calcular el error de reconstrucción cuadrático
    reconstruction_error = torch.nn.MSELoss()(y_pred, y_true)

    # Calcular la penalización de las condiciones de borde
    border_penalty = torch.sum((y_pred[:, :, 0, :] - input_data[:, :, 0, :])**2)
    border_penalty += torch.sum((y_pred[:, :, -1, :] - input_data[:, :, -1, :])**2)
    border_penalty += torch.sum((y_pred[:, 0, :, :] - input_data[:, 0, :, :])**2)
    border_penalty += torch.sum((y_pred[:, -1, :, :] - input_data[:, -1, :, :])**2)

    # Combinar los dos componentes de la pérdida
    loss = reconstruction_error + border_penalty

    return loss

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
# Definir el dispositivo en el que se entrenará la red
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Mover la red al dispositivo
net = CNN().to(device)

# Definir el optimizador
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Definir el número de épocas
num_epochs = 2000

# Bucle de entrenamiento
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):  # Ahora i es el índice de iteración
        # Mover los datos al dispositivo
        x = x.to(device)
        y = y.to(device)

        # Pasar los datos a través de la red
        output = net(x)

        # Calcular la pérdida
        loss = custom_loss(output, y, x)

        # Retropropagar el error y actualizar los pesos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Imprimir la pérdida cada 100 épocas
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss = {loss.item()}')

# Mover la red al modo de evaluación
net.eval()

# Inicializar la pérdida total
total_loss = 0.0

# Bucle de prueba
with torch.no_grad():  # No necesitamos calcular gradientes en el bucle de prueba
    for i, (x, y) in enumerate(test_loader):
        # Mover los datos al dispositivo
        x = x.to(device)
        y = y.to(device)

        # Pasar los datos a través de la red
        output = net(x)

        # Calcular la pérdida
        loss = custom_loss(output, y, x)

        # Acumular la pérdida total
        total_loss += loss.item()

# Calcular la pérdida media
average_loss = total_loss / len(test_loader)

print(f'Test Loss = {average_loss}')
