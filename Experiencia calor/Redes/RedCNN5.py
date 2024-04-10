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

shutil.rmtree("runs")
mat_fname = "Datasets/mi_matriz_solo_diritletch"
mat = sio.loadmat(mat_fname)
matriz_cargada = mat["dataset_matriz"]


num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]

total_casos = matriz_cargada_mezclada.shape[0]
porcentaje_entrenamiento = 0.7
num_entrenamiento = int(total_casos * porcentaje_entrenamiento)
num_pruebas = total_casos - num_entrenamiento

temp_dirichlet_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 12]
temp_dirichlet_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 12]

temp_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 17]
temp_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 17]

temp_dirichlet_train_tensor = torch.from_numpy(temp_dirichlet_train).float()
temp_dirichlet_test_tensor = torch.from_numpy(temp_dirichlet_test).float()
temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()


temp_dirichlet_train_tensor = temp_dirichlet_train_tensor.unsqueeze(1)
temp_dirichlet_test_tensor = temp_dirichlet_test_tensor.unsqueeze(1)

temp_train_tensor = temp_train_tensor.unsqueeze(1)
temp_test_tensor = temp_test_tensor.unsqueeze(1)

train_dataset = TensorDataset(temp_train_tensor, temp_train_tensor)
test_dataset = TensorDataset(temp_test_tensor, temp_test_tensor)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class HeatPropagationNet(nn.Module):
    def __init__(self):
        super(HeatPropagationNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

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

    for output in outputs:
        batch_size, num_channels, _, _ = output.shape

        for i in range(num_channels):
            feature_map = output[:, i, :, :].unsqueeze(1)

            border_loss = F.mse_loss(
                feature_map[:, :, :, [0, -1]], target[:, :, :, [0, -1]]
            ) + F.mse_loss(feature_map[:, :, [0, -1], :], target[:, :, [0, -1], :])
            loss_borde_total += border_loss

            if i == num_channels - 1:
                interior_loss = F.mse_loss(
                    feature_map[..., 1:-1, 1:-1], target[..., 1:-1, 1:-1]
                )
                loss_interior_total += interior_loss

            num_feature_maps += 1

    loss_borde_total /= num_feature_maps
    loss_interior_total /= num_feature_maps

    return (
        ponderacion_frontera * loss_borde_total
        + ponderacion_interior * loss_interior_total
    )


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

writer = SummaryWriter("runs/experiment_1")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = HeatPropagationNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


ponderacion_interior = 0.5
ponderacion_frontera = 0.5


for epoch in range(1000):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs_all = outputs[:-1]
        output_final = outputs[-1:]
        loss_all = custom_loss(
            outputs_all, labels, ponderacion_interior, ponderacion_frontera
        )
        loss_final = custom_loss(
            output_final, labels, ponderacion_interior, ponderacion_frontera
        )

        loss_all.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoca {epoch}, Toda la loss: {loss_all.item()}")
        print(f"Epoca {epoch}, Toda ultima capa: {loss_final.item()}")

    writer.add_scalar("Loss_all_layers", loss_all.item(), epoch)
    writer.add_scalar("Loss_final_layer", loss_final.item(), epoch)

    feature_map1_single = outputs[0][0, :3].unsqueeze(0)
    feature_map2_single = outputs[1][0, :3].unsqueeze(0)
    feature_map3_single = outputs[2][0, :3].unsqueeze(0)

    feature_map1_single = (feature_map1_single - feature_map1_single.min()) / (
        feature_map1_single.max() - feature_map1_single.min()
    )
    feature_map2_single = (feature_map2_single - feature_map2_single.min()) / (
        feature_map2_single.max() - feature_map2_single.min()
    )
    feature_map3_single = (feature_map3_single - feature_map3_single.min()) / (
        feature_map3_single.max() - feature_map3_single.min()
    )

    writer.add_images("Feature Maps 1", feature_map1_single, epoch)
    writer.add_images("Feature Maps 2", feature_map2_single, epoch)
    writer.add_images("Feature Maps 3", feature_map3_single, epoch)

    inputs_normalized = (inputs - inputs.min()) / (inputs.max() - inputs.min())
    labels_normalized = (labels - labels.min()) / (labels.max() - labels.min())

    writer.add_images("Input Images", inputs_normalized, epoch)
    writer.add_images("Ground Truth", labels_normalized, epoch)
    """
    kernels = model.conv1.weight.detach().clone()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()


    if kernels.shape[1] == 1:

        kernels = torch.cat((kernels, kernels, kernels), 1)


    writer.add_images('conv1 Kernels', kernels, epoch)
    """

writer.close()
print("Cambiando a evaluacion...")
model.eval()

total_loss = 0
num_batches = 0

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = custom_loss(outputs, labels, ponderacion_interior, ponderacion_frontera)
        total_loss += loss.item()
        num_batches += 1


mean_loss = total_loss / num_batches

print("Mean Loss:", mean_loss)
