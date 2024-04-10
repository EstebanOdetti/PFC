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


mat_fname = 'Datasets/mi_matriz.mat'
mat_fname_EO = 'Datasets/mi_matriz_conmuta_este_oeste.mat'
mat = sio.loadmat(mat_fname)
mat_EO = sio.loadmat(mat_fname_EO)
matriz_cargada = mat['dataset_matriz']
matriz_cargada_EO = mat_EO['dataset_matriz']



num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]

train = matriz_cargada_mezclada[0:137,:,:,0:16]
test = matriz_cargada_mezclada[138:,:,:,0:16]

train_EO = matriz_cargada_EO[0:137,:,:,0:16]
test_EO = matriz_cargada_EO[138:,:,:,0:16]

temp_train = matriz_cargada_mezclada[0:137,:,:,17]
temp_test = matriz_cargada_mezclada[138:,:,:,17]

temp_train_EO = matriz_cargada_EO[0:137,:,:,17]
temp_test_EO = matriz_cargada_EO[138:,:,:,17]
print(train.shape)
train_tensor = torch.from_numpy(train).float()
test_tensor = torch.from_numpy(test).float()

train_tensor_EO = torch.from_numpy(train_EO).float()
test_tensor_EO = torch.from_numpy(test_EO).float()

temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()

temp_train_tensor_EO = torch.from_numpy(temp_train_EO).float()
temp_test_tensor_EO = torch.from_numpy(temp_test_EO).float()



mean_train = torch.mean(train_tensor)
std_train = torch.std(train_tensor)


train_tensor_normalized = (train_tensor - mean_train) / std_train
test_tensor_normalized = (test_tensor - mean_train) / std_train


mean_train_EO = torch.mean(train_tensor_EO)
std_train_EO = torch.std(train_tensor_EO)


train_tensor_EO_normalized = (train_tensor_EO - mean_train_EO) / std_train_EO
test_tensor_EO_normalized = (test_tensor_EO - mean_train_EO) / std_train_EO


mean_temp_train = torch.mean(temp_train_tensor)
std_temp_train = torch.std(temp_train_tensor)


temp_train_tensor_normalized = (temp_train_tensor - mean_temp_train) / std_temp_train
temp_test_tensor_normalized = (temp_test_tensor - mean_temp_train) / std_temp_train


mean_temp_train_EO = torch.mean(temp_train_tensor_EO)
std_temp_train_EO = torch.std(temp_train_tensor_EO)


temp_train_tensor_EO_normalized = (temp_train_tensor_EO - mean_temp_train_EO) / std_temp_train_EO
temp_test_tensor_EO_normalized = (temp_test_tensor_EO - mean_temp_train_EO) / std_temp_train_EO



train_dataset = TensorDataset(train_tensor.permute(0, 3, 1, 2), temp_train_tensor)

train_dataset_EO = TensorDataset(train_tensor_EO.permute(0, 3, 1, 2), temp_train_tensor_EO)

test_dataset = TensorDataset(test_tensor.permute(0, 3, 1, 2), temp_test_tensor)

test_dataset_EO = TensorDataset(test_tensor_EO.permute(0, 3, 1, 2), temp_test_tensor_EO)

train_dataset_normalized = TensorDataset(train_tensor_normalized.permute(0, 3, 1, 2), temp_train_tensor_normalized)

train_dataset_EO_normalized = TensorDataset(train_tensor_EO_normalized.permute(0, 3, 1, 2), temp_train_tensor_EO_normalized)

test_dataset_normalized = TensorDataset(test_tensor_normalized.permute(0, 3, 1, 2), temp_test_tensor_normalized)

test_dataset_EO_normalized = TensorDataset(test_tensor_EO_normalized.permute(0, 3, 1, 2), temp_test_tensor_EO_normalized)


batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_loader_EO = DataLoader(train_dataset_EO, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

test_loader_EO = DataLoader(test_dataset_EO, batch_size=batch_size, shuffle=True)

train_loader_normalized = DataLoader(train_dataset_normalized, batch_size=batch_size, shuffle=True)

train_loader_EO_normalized = DataLoader(train_dataset_EO_normalized, batch_size=batch_size, shuffle=True)

test_loader_normalized = DataLoader(test_dataset_normalized, batch_size=batch_size, shuffle=True)

test_loader_EO_normalized = DataLoader(test_dataset_EO_normalized, batch_size=batch_size, shuffle=True)



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
    
device = torch.device('cuda:0')
net=CNN()
net = net.to(device)

learning_rate = 0.0001
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
criterion = torch.nn.MSELoss() 
num_epochs = 1500
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
            print(f'Epoch {epoch}, Loss = {loss}')


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


        mse = mean_squared_error(real_output.cpu().detach().numpy(), predicted_output.cpu().detach().numpy())
        mse_list.append(mse)

    print(f'MSE on test data: {np.mean(mse_list)}')



with torch.no_grad():
    mse_list = []

    for i in range(len(test_tensor_EO)):
        inputs = test_tensor_EO[i]
        real_output = temp_test_tensor_EO[i]

        inputs = inputs.permute(2, 0, 1)
        inputs = inputs.to(device)
        real_output = real_output.to(device)



        predicted_output = net(inputs.unsqueeze(0)).squeeze()


        mse = mean_squared_error(real_output.cpu().detach().numpy(), predicted_output.cpu().detach().numpy())
        mse_list.append(mse)

    print(f'MSE on test data: {np.mean(mse_list)}')



net_normalized=CNN()
net_normalized = net.to(device)

learning_rate = 0.0001
optimizer=torch.optim.Adam(net_normalized.parameters(),lr=learning_rate)
criterion = torch.nn.MSELoss() 
num_epochs = 1500
loss_list = []


first_tensor = train_loader_normalized.dataset.tensors[0]

print(first_tensor.size())

for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader_normalized):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        loss = criterion(net_normalized(x), y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss = {loss}')


net.eval()

with torch.no_grad():
    mse_list = []

    for i in range(len(test_tensor_normalized)):
        inputs = test_tensor_normalized[i]
        real_output = temp_test_tensor_normalized[i]

        inputs = inputs.permute(2, 0, 1)
        inputs = inputs.to(device)
        real_output = real_output.to(device)



        predicted_output = net(inputs.unsqueeze(0)).squeeze()


        mse = mean_squared_error(real_output.cpu().detach().numpy(), predicted_output.cpu().detach().numpy())
        mse_list.append(mse)

    print(f'MSE on test data: {np.mean(mse_list)}')



with torch.no_grad():
    mse_list = []

    for i in range(len(test_tensor_EO_normalized)):
        inputs = test_tensor_EO_normalized[i]
        real_output = temp_test_tensor_EO_normalized[i]

        inputs = inputs.permute(2, 0, 1)
        inputs = inputs.to(device)
        real_output = real_output.to(device)



        predicted_output = net(inputs.unsqueeze(0)).squeeze()


        mse = mean_squared_error(real_output.cpu().detach().numpy(), predicted_output.cpu().detach().numpy())
        mse_list.append(mse)

    print(f'MSE on test data: {np.mean(mse_list)}')