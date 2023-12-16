import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Leer y preparar el conjunto de datos
file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/dataset_avanzado_random_reordenado.csv'
data = pd.read_csv(file_path, header=None)
data.columns = [
    'front_wheel_1', 'front_wheel_2', 'front_wheel_3', 'front_wheel_4',
    'rear_wheel_1', 'rear_wheel_2', 'rear_wheel_3', 'rear_wheel_4',
    'front_target_1', 'front_target_2', 'rear_target_1', 'rear_target_2'
]
front_wheel_data = data[['front_wheel_1', 'front_wheel_2', 'front_wheel_3', 'front_wheel_4']].to_numpy()
rear_wheel_data = data[['rear_wheel_1', 'rear_wheel_2', 'rear_wheel_3', 'rear_wheel_4']].to_numpy()
front_targets = data[['front_target_1', 'front_target_2']].to_numpy()[::30]
rear_targets = data[['rear_target_1', 'rear_target_2']].to_numpy()[::30]

front_wheel_data = front_wheel_data.reshape(-1, 30, 4)
rear_wheel_data = rear_wheel_data.reshape(-1, 30, 4)

# Convertir los datos a tensores de PyTorch
front_wheel_data = torch.tensor(front_wheel_data, dtype=torch.float32).unsqueeze(1)
rear_wheel_data = torch.tensor(rear_wheel_data, dtype=torch.float32).unsqueeze(1)
front_targets = torch.tensor(front_targets, dtype=torch.float32)
rear_targets = torch.tensor(rear_targets, dtype=torch.float32)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Capas convolucionales de la rueda delantera
        self.front_wheel_conv1 = nn.Conv2d(1, 32, 3, padding=1)        
        self.front_wheel_pool1 = nn.MaxPool2d(2, 2)  # Agregamos capa de pooling
        self.front_wheel_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.front_wheel_pool2 = nn.MaxPool2d(2, 2)  # Agregamos capa de pooling

        # Capas convolucionales de la rueda trasera
        self.rear_wheel_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.rear_wheel_pool1 = nn.MaxPool2d(2, 2)  # Agregamos capa de pooling
        self.rear_wheel_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.rear_wheel_pool2 = nn.MaxPool2d(2, 2)  # Agregamos capa de pooling

        # Capa convolucional combinada
        self.combined_conv = nn.Conv2d(128, 1, 3, padding=1)

        # Aplanamiento y capa completamente conectada
        # NOTA: El tamaño de entrada de la capa lineal es un valor de ejemplo, 
        # tendrás que calcular el tamaño correcto basado en la salida de tus capas convolucionales y de pooling.
        self.fc = nn.Linear(7, 2)  # 7 características de entrada, 2 de salida  

    def forward(self, front_wheel_data, rear_wheel_data):
        # Pasar los datos a través de las capas de la rueda delantera con activación ReLU
        front_wheel_data = F.relu(self.front_wheel_conv1(front_wheel_data))
        front_wheel_data = self.front_wheel_pool1(front_wheel_data)
        front_wheel_data = F.relu(self.front_wheel_conv2(front_wheel_data))
        front_wheel_data = self.front_wheel_pool2(front_wheel_data)

        # Pasar los datos a través de las capas de la rueda trasera con activación ReLU
        rear_wheel_data = F.relu(self.rear_wheel_conv1(rear_wheel_data))
        rear_wheel_data = self.rear_wheel_pool1(rear_wheel_data)
        rear_wheel_data = F.relu(self.rear_wheel_conv2(rear_wheel_data))
        rear_wheel_data = self.rear_wheel_pool2(rear_wheel_data)

        # Combinar y pasar por la capa convolucional restante con activación ReLU
        combined = torch.cat([front_wheel_data, rear_wheel_data], dim=1)
        combined = F.relu(self.combined_conv(combined))

        # Aplanar y pasar a través de la capa completamente conectada con activación ReLU
        combined = combined.view(combined.size(0), -1)  # Aplanamiento correcto para la entrada a la capa lineal
        output = self.fc(combined)
        return output
# Crear el modelo
model = CNNModel()

# Definir el criterio de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Crear DataLoader
dataset = TensorDataset(front_wheel_data, rear_wheel_data, front_targets, rear_targets)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Entrenar el modelo
n_epochs = 1000
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (front_wheel_data, rear_wheel_data, front_targets, rear_targets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Calcular la salida del modelo para ambos objetivos
        outputs_front = model(front_wheel_data, rear_wheel_data)
        
        # Calcular la pérdida para ambos objetivos
        loss_front = criterion(outputs_front, front_targets)
        
        # Retropropagar el error y actualizar los pesos del modelo
        loss_front.backward()
        optimizer.step()
        
        running_loss += loss_front.item()
    
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

# Evaluar el modelo
model.eval()
with torch.no_grad():
    outputs_front = model(front_wheel_data, rear_wheel_data)
    loss_front = criterion(outputs_front, front_targets)
    print('Loss: ', loss_front.item())