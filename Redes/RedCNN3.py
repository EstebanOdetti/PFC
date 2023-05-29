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
from torch.utils.tensorboard import SummaryWriter

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
temp_dirichlet_train[:, 1:-1, 1:-1] = 1
temp_dirichlet_test[:, 1:-1, 1:-1] = 1
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3,3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), stride=1, padding=1)
        #self.dropout = nn.Dropout2d(p=0.2)  # Añade una capa de Dropout. 'p' es la probabilidad de que cada nodo se apague.

    def forward(self, x):
        #x1 = self.dropout(F.leaky_relu(self.conv1(x)))
        x1 = F.relu(self.conv1(x))
        #x2 = self.dropout(F.leaky_relu(self.conv2(x1)))
        x2 = F.relu(self.conv2(x1))
        #x3 = self.dropout(F.leaky_relu(self.conv3(x2)))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = self.conv6(x5)  # Normalmente no aplicamos Dropout en la capa de salida
        return x1, x2, x3, x4, x5, x6


def custom_loss(outputs, target, ponderacion_interior, ponderacion_frontera):
    loss_borde_total = 0
    loss_interior_total = 0
    num_feature_maps = 0

    for output in outputs:  # Recorre todas las salidas de capa
        batch_size, num_channels, _, _ = output.shape  # Obtén el número de canales en esta salida de capa

        for i in range(num_channels):  # Recorre cada mapa de características
            feature_map = output[:, i, :, :].unsqueeze(1)  # Selecciona el mapa de características y añade una dimensión de canal

            # Calcula la pérdida en los bordes
            border_loss = F.mse_loss(feature_map[:, :, :, [0, -1]], target[:, :, :, [0, -1]]) + \
                          F.mse_loss(feature_map[:, :, [0, -1], :], target[:, :, [0, -1], :])
            loss_borde_total += border_loss

            # Calcula la pérdida en el interior solo para la última capa
            if i == num_channels - 1:
                interior_loss = F.mse_loss(feature_map[..., 1:-1, 1:-1], target[..., 1:-1, 1:-1])
                loss_interior_total += interior_loss

            num_feature_maps += 1

    # Normaliza las pérdidas
    loss_borde_total /= num_feature_maps
    loss_interior_total /= num_feature_maps

    return ponderacion_frontera * loss_borde_total + ponderacion_interior * loss_interior_total


def plot_feature_maps(feature_maps, num_cols=6):
    num_kernels = feature_maps.shape[1]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(feature_maps[0, i].cpu().detach().numpy())
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    
def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i].cpu().numpy())
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# Verifica si CUDA está disponible y selecciona el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializar la red y el optimizador
model = HeatPropagationNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Parámetros para la función de pérdida
ponderacion_interior = 0.1
ponderacion_frontera = 0.9

# Bucle de entrenamiento
for epoch in range(10000):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Cero los gradientes del optimizador
        optimizer.zero_grad()

        # Pase adelante, cálculo de la pérdida, pase atrás y optimización
        feature_maps1, feature_maps2, feature_maps3, feature_maps4, feature_maps5, output = model(inputs)
        outputs = [feature_maps1, feature_maps2, feature_maps3,feature_maps4, feature_maps5, output]
        outputs_2 = [output]
        loss = custom_loss(outputs, labels, ponderacion_interior, ponderacion_frontera)
        loss_2 = custom_loss(outputs_2, labels, ponderacion_interior, ponderacion_frontera)
        loss.backward()
        optimizer.step()

    # Imprime estadísticas de entrenamiento
    print('Epoch: %d, Loss_total: %.3f' % (epoch + 1, loss.item()))
    print('Epoch: %d, Loss_ultima_capa: %.3f' % (epoch + 1, loss_2.item()))

    # Visualizar los mapas de características después de cada época
    #if epoch % 100 == 0:  # ajusta este número para visualizar los mapas de características con la frecuencia que desees
        #plot_feature_maps(feature_maps1)
        #plot_feature_maps(feature_maps2)
        #plot_feature_maps(feature_maps3)

        # Visualizar los kernels de las capas convolucionales
        #kernels1 = model.conv1.weight.detach().clone()
        #kernels1 = kernels1 - kernels1.min()
        #kernels1 = kernels1 / kernels1.max()
        #plot_kernels(kernels1)

        #kernels2 = model.conv2.weight.detach().clone()
        #kernels2 = kernels2 - kernels2.min()
        #kernels2 = kernels2 / kernels2.max()
        #plot_kernels(kernels2)

        #kernels3 = model.conv3.weight.detach().clone()
        #kernels3 = kernels3 - kernels3.min()
        #kernels3 = kernels3 / kernels3.max()
        #plot_kernels(kernels3)

