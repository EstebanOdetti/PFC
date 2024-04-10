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
mat_fname = "Datasets/mi_matriz_solo_diritletch_enriquesida.mat"
mat = sio.loadmat(mat_fname)
matriz_cargada = mat["dataset_matriz"]


num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]

train = matriz_cargada_mezclada[0:137, :, :, 0:16]
test = matriz_cargada_mezclada[138:, :, :, 0:16]
temp_train = matriz_cargada_mezclada[0:137, :, :, 17]
temp_test = matriz_cargada_mezclada[138:, :, :, 17]
print(train.shape)
train_tensor = torch.from_numpy(train).float()
test_tensor = torch.from_numpy(test).float()
temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()

primeros_10_casos = temp_train_tensor[0:10]
for i in range(10):
    caso = primeros_10_casos[i]
    imagen = caso[:, :]
    plt.subplot(2, 5, i + 1)
    im = plt.imshow(caso, cmap="hot", origin="lower")
    plt.title(f"Caso {i+1}")


plt.tight_layout(rect=[0, 0, 0.9, 1])


cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label("Temperatura", rotation=270, labelpad=15)

plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 2 * 2, 49)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 7, 7)
        return x


device = torch.device("cuda:0")
net = CNN()
net = net.to(device)

learning_rate = 0.0001


optimmizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

criterion = torch.nn.MSELoss()

num_epochs = 100
loss_list = []
epoch_loss_list = []

for i in range(num_epochs):
    total_loss = 0.0
    for j in range(136):
        optimmizer.zero_grad()
        x = train_tensor[j]
        y = temp_train_tensor[j]

        x = x.permute(2, 0, 1)

        x = x.to(device)
        y = y.to(device)

        loss = criterion(net(x.unsqueeze(0)), y)
        loss.backward()
        optimmizer.step()

        valor_perdida = loss.data.item()
        loss_list.append(valor_perdida)
        total_loss += valor_perdida

        print(f"Época {i}, Lote {j}, Pérdida = {valor_perdida}")

    perdida_promedio = total_loss / 136
    epoch_loss_list.append(perdida_promedio)
    print(f"Época {i}, Pérdida Promedio = {perdida_promedio}")


plt.figure(figsize=(10, 6))
plt.plot(epoch_loss_list)
plt.title("Pérdida por Época")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.show()

net.eval()


predictions = []


with torch.no_grad():
    total_loss = 0.0
    num_samples = 0
    for j in range(len(test_tensor)):
        x_test = test_tensor[j]
        y_test = temp_test_tensor[j]

        x_test = x_test.permute(2, 0, 1)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        output = net(x_test.unsqueeze(0))
        predictions.append(output.squeeze().cpu().numpy())

        loss = criterion(output.squeeze(), y_test)
        total_loss += loss.item()
        num_samples += 1


mean_loss = total_loss / num_samples
print(f"Pérdida media en el conjunto de test = {mean_loss}")


predictions_tensor = torch.tensor(predictions)

import matplotlib.pyplot as plt
import torch


primeros_10_casos = temp_train_tensor[0:10]
primeras_10_predicciones = predictions_tensor[0:10]

plt.figure(figsize=(15, 10))

for i in range(10):

    caso = primeros_10_casos[i]
    imagen_caso = caso.numpy() if isinstance(caso, torch.Tensor) else caso
    plt.subplot(10, 2, 2 * i + 1)
    im = plt.imshow(imagen_caso, cmap="hot", origin="lower")
    plt.title(f"Caso {i+1} - Ground Truth")

    prediccion = primeras_10_predicciones[i]
    imagen_prediccion = (
        prediccion.numpy() if isinstance(prediccion, torch.Tensor) else prediccion
    )
    plt.subplot(10, 2, 2 * i + 2)
    plt.imshow(imagen_prediccion, cmap="hot", origin="lower")
    plt.title(f"Caso {i+1} - Predicción")


plt.show()
