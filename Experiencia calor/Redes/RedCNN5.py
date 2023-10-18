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
from sklearn.metrics import r2_score
import shutil 
#Borramos los runs anteriores
shutil.rmtree('runs')
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
train_dataset = TensorDataset(temp_train_tensor, temp_train_tensor)
test_dataset = TensorDataset(temp_test_tensor, temp_test_tensor)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
class HeatPropagationNet(nn.Module):
    def __init__(self):
        super(HeatPropagationNet, self).__init__()
        # Bloque de entrada
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Bloque intermedio
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # Bloque de salida
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu3(self.conv3(self.relu2(self.conv2(x1))))
        x3 = self.conv4(x2)
        return x3, x2, x1

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


#plt.ion()  # Habilita el modo interactivo

# Crea las figuras y los ejes que se actualizarán durante el entrenamiento
#fig_ground_truth, ax_ground_truth = plt.subplots(1, 2)
#fig_feature_map, ax_feature_map = plt.subplots()
"""
def show_ground_truth(img, label, fig, ax):
    img = img.cpu().numpy().squeeze()
    label = label.cpu().numpy().squeeze()

    ax[0].imshow(img)
    ax[0].title.set_text('Input Image')
    ax[1].imshow(label)
    ax[1].title.set_text('Ground Truth')
    
    fig.canvas.draw()  
    fig.canvas.flush_events()

def plot_feature_maps(feature_maps, fig, ax, num_cols=6):
    num_kernels = feature_maps.shape[1]
    num_rows = 1+ num_kernels // num_cols
    fig.set_size_inches(num_cols,num_rows)
    fig.clf()
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(feature_maps[0, i].cpu().detach().numpy())
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.canvas.draw()  
    fig.canvas.flush_events()
    
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
"""
# Inicializa el escritor de TensorBoard
writer = SummaryWriter('runs/experiment_1')

# Verifica si CUDA está disponible y selecciona el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializar la red y el optimizador
model = HeatPropagationNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Parámetros para la función de pérdida
ponderacion_interior = 0.5
ponderacion_frontera = 0.5

# Bucle de entrenamiento
for epoch in range(1000):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Cero los gradientes del optimizador
        optimizer.zero_grad()

        # Pase adelante, cálculo de la pérdida, pase atrás y optimización
        outputs = model(inputs)
        outputs_all = outputs[:-1]
        output_final = outputs[-1:]
        loss_all = custom_loss(outputs_all, labels, ponderacion_interior, ponderacion_frontera)
        loss_final = custom_loss(output_final, labels, ponderacion_interior, ponderacion_frontera)
        
        # Backward and optimization
        loss_all.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoca {epoch}, Toda la loss: {loss_all.item()}')
        print(f'Epoca {epoch}, Toda ultima capa: {loss_final.item()}')


    # Añade las pérdidas a TensorBoard
    writer.add_scalar('Loss_all_layers', loss_all.item(), epoch)
    writer.add_scalar('Loss_final_layer', loss_final.item(), epoch)

    # Selecciona los primeros 3 mapas de características de cada capa y añade una dimensión de batch
    feature_map1_single = outputs[0][0, :3].unsqueeze(0)
    feature_map2_single = outputs[1][0, :3].unsqueeze(0)
    feature_map3_single = outputs[2][0, :3].unsqueeze(0)

    # Asegura que los valores estén en el rango [0, 1]
    feature_map1_single = (feature_map1_single - feature_map1_single.min()) / (feature_map1_single.max() - feature_map1_single.min())
    feature_map2_single = (feature_map2_single - feature_map2_single.min()) / (feature_map2_single.max() - feature_map2_single.min())
    feature_map3_single = (feature_map3_single - feature_map3_single.min()) / (feature_map3_single.max() - feature_map3_single.min())

    writer.add_images('Feature Maps 1', feature_map1_single, epoch)
    writer.add_images('Feature Maps 2', feature_map2_single, epoch)
    writer.add_images('Feature Maps 3', feature_map3_single, epoch)

    # Normaliza las entradas y las etiquetas
    inputs_normalized = (inputs - inputs.min()) / (inputs.max() - inputs.min())
    labels_normalized = (labels - labels.min()) / (labels.max() - labels.min())

    writer.add_images('Input Images', inputs_normalized, epoch)
    writer.add_images('Ground Truth', labels_normalized, epoch)
    """
    kernels = model.conv1.weight.detach().clone()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()

    # Comprobar si los kernels tienen un solo canal
    if kernels.shape[1] == 1:
        # Triplicar los canales
        kernels = torch.cat((kernels, kernels, kernels), 1)

    # Ahora puedes agregar los kernels a TensorBoard
    writer.add_images('conv1 Kernels', kernels, epoch)
    """

writer.close()
print("Cambiando a evaluacion...")
model.eval()  # Cambia el modo del modelo a evaluación

total_loss = 0
num_batches = 0

with torch.no_grad():  # No necesitamos calcular gradientes en el test
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Haz una predicción con el modelo
        outputs = model(inputs)

        # Calcula la pérdida
        loss = custom_loss(outputs, labels, ponderacion_interior, ponderacion_frontera)
        total_loss += loss.item()
        num_batches += 1
        

# Calcula las métricas medias
mean_loss = total_loss / num_batches

print('Mean Loss:', mean_loss)


