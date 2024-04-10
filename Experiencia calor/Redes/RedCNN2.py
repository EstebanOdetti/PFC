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


def loss_fn(model, Dataset):

    x.requires_grad = True

    phi_hat = model(x)

    grad_phi = grad(
        phi_hat, x, grad_outputs=torch.ones_like(phi_hat), create_graph=True
    )[0]

    laplace_phi = torch.zeros_like(phi_hat)
    for i in range(x.shape[-1]):
        grad_phi_i = grad_phi[:, i]
        grad_grad_phi_i = grad(
            grad_phi_i, x, grad_outputs=torch.ones_like(grad_phi_i), create_graph=True
        )[0]
        laplace_phi += grad_grad_phi_i[:, i]

    loss_pde = torch.mean(((k * laplace_phi) + Q) ** 2)

    phi_bc_pred = model(x_bc)
    loss_bc = torch.mean((phi_bc - phi_bc_pred) ** 2)

    loss = loss_pde + loss_bc

    return loss


mat_fname = "Datasets/mi_matriz.mat"
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

primeros_10_casos = temp_train_tensor[34:44]
for i in range(10):
    caso = primeros_10_casos[i]
    imagen = caso[:, :]
    plt.subplot(2, 5, i + 1)
    plt.imshow(imagen, cmap="gray")
    plt.axis("off")
    plt.title(f"Caso {i+1}")
plt.tight_layout()
plt.show()


train_dataset = TensorDataset(train_tensor.permute(0, 3, 1, 2), temp_train_tensor)
test_dataset = TensorDataset(test_tensor.permute(0, 3, 1, 2), temp_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


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
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
num_epochs = 2000
loss_list = []


first_tensor = train_loader.dataset.tensors[0]

print(first_tensor.size())

for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        loss = criterion(net(x), y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {loss}")


net.eval()

with torch.no_grad():
    mse_list = []

    for i in range(len(test_tensor)):
        inputs = test_tensor[i]
        real_output = temp_test_tensor[i]

        inputs = inputs.permute(2, 0, 1)
        inputs = inputs.to(device)
        real_output = real_output.to(device)

        predicted_output = net(inputs.unsqueeze(0)).squeeze()

        mse = mean_squared_error(
            real_output.cpu().detach().numpy(), predicted_output.cpu().detach().numpy()
        )
        mse_list.append(mse)

    print(f"MSE on test data: {np.mean(mse_list)}")
