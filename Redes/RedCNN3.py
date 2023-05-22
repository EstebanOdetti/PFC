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
total_casos = matriz_cargada_mezclada.shape[0]
porcentaje_entrenamiento = 0.7
num_entrenamiento = int(total_casos * porcentaje_entrenamiento)
num_pruebas = total_casos - num_entrenamiento
#obtenermos las temperaturas en los bordes y en el interior es cero (Y ESTO ESS LO Q LE ENTRA A LA RED)
temp_dirichlet_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 12]
temp_dirichlet_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 12]
# esto es el ground truth. 
temp_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 17]
temp_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 17]
#convertis en tensores
temp_dirichlet_train_tensor = torch.from_numpy(temp_dirichlet_train).float()
temp_dirichlet_test_tensor = torch.from_numpy(temp_dirichlet_test).float()
temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()
#mostras los 10 primeros casos
#primeros_10_casos = temp_train_tensor[0:10]
#for i in range(10):
#    caso = primeros_10_casos[i]
#    imagen = caso[:, :]
#    plt.subplot(2, 5, i + 1)
#    plt.imshow(imagen, cmap='hot')  # Utilizar cmap='hot' para representar temperaturas
#    plt.title(f'Caso {i+1}')
#plt.tight_layout()
#plt.show()
# Graficar los primeros 10 casos de solo borde
#primeros_10_casos = temp_dirichlet_train_tensor[0:10]
#for i in range(primeros_10_casos.shape[0]):
#    caso = primeros_10_casos[i]
#    imagen = caso  # No se necesita [:, :]
#    plt.subplot(2, 5, i + 1)
#    plt.imshow(imagen, cmap='hot')
#    plt.title(f'Caso {i+1}')
#plt.tight_layout()
#plt.show()
# Añadir una dimensión extra para los canales
temp_dirichlet_train_tensor = temp_dirichlet_train_tensor.unsqueeze(1) # tamaño ahora es [num_entrenamiento, 1, height, width]
temp_dirichlet_test_tensor = temp_dirichlet_test_tensor.unsqueeze(1) # tamaño ahora es [num_pruebas, 1, height, width]
# Añadir una dimensión extra para los canales a los targets si es necesario
temp_train_tensor = temp_train_tensor.unsqueeze(1)
temp_test_tensor = temp_test_tensor.unsqueeze(1)
# Crear TensorDatasets y DataLoaders
train_dataset = TensorDataset(temp_dirichlet_train_tensor, temp_train_tensor)
test_dataset = TensorDataset(temp_dirichlet_test_tensor, temp_test_tensor)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
class HeatPropagationNet(nn.Module):
    def __init__(self):
        super(HeatPropagationNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(p=0.2)  # Añade una capa de Dropout. 'p' es la probabilidad de que cada nodo se apague.

    def forward(self, x):
        x1 = self.dropout(F.leaky_relu(self.conv1(x)))
        x2 = self.dropout(F.leaky_relu(self.conv2(x1)))
        x3 = self.dropout(F.leaky_relu(self.conv3(x2)))
        x4 = self.conv4(x3)  # Normalmente no aplicamos Dropout en la capa de salida
        return x1, x2, x3, x4


def custom_loss(outputs, target, ponderacion_interior, ponderacion_frontera):
    loss_borde = 0
    loss_interior = 0
    for i, output in enumerate(outputs):
        # calculamos la pérdida en los bordes sumando todos los canales
        border_loss = F.mse_loss(output, target)
        loss_borde += border_loss

        # calculamos la pérdida en el interior solo para la última capa
        if i == len(outputs) - 1:
            interior_loss = F.mse_loss(output[..., 1:-1, 1:-1], target[..., 1:-1, 1:-1])
            loss_interior += interior_loss

    loss_borde /= len(outputs)  # Normalizamos la pérdida en los bordes

    return ponderacion_frontera * loss_borde + ponderacion_interior * loss_interior


# Verifica si CUDA está disponible y selecciona el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializar la red y el optimizador
model = HeatPropagationNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Parámetros para la función de pérdida
ponderacion_interior = 0.6
ponderacion_frontera = 0.4

# Bucle de entrenamiento
for epoch in range(10000):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Cero los gradientes del optimizador
        optimizer.zero_grad()

        # Pase adelante, cálculo de la pérdida, pase atrás y optimización
        outputs = model(inputs)
        loss = custom_loss(outputs, labels, ponderacion_interior, ponderacion_frontera)
        loss.backward()
        optimizer.step()

    # Imprime estadísticas de entrenamiento
    print('Epoch: %d, Loss: %.3f' % (epoch + 1, loss.item()))
