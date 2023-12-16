import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Directorio base donde se encuentran tus archivos FFT y GMM
directorio_base = os.path.dirname(__file__)  # Obtener el directorio del script actual

# Leer y preparar el conjunto de datos
file_path = os.path.join(directorio_base, 'Datasets', 'dataset_avanzado_random_en_lista.csv')
data = pd.read_csv(file_path, header=None)
data.columns = [
    'front_wheel_freq', 'front_wheel_psdx', 'front_wheel_psdy', 'front_wheel_psdz',
    'front_target_freq', 'front_target_ten'
]
wheel_data = data[['front_wheel_freq', 'front_wheel_psdx', 'front_wheel_psdy', 'front_wheel_psdz']].to_numpy()
targets = data[['front_target_freq']].to_numpy()[::30]  # Seleccionar solo la penúltima columna

wheel_data = wheel_data.reshape(-1, 30, 4)

# Convertir los datos a tensores de PyTorch
wheel_data = torch.tensor(wheel_data, dtype=torch.float32).unsqueeze(1)
targets = torch.tensor(targets, dtype=torch.float32)

# Definir el modelo
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.combined_conv = nn.Conv2d(64, 1, (3, 3), padding=1)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(120, 1)  # Salida de una sola dimensión (Dimensión 1)

    def forward(self, wheel_data):
        x = self.conv1(wheel_data)
        x = self.conv2(x)
        x = self.combined_conv(x)
        x = self.flatten(x)
        output = self.fc(x)
        return output

# Crear el modelo
model = CNNModel()

# Definir el criterio de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Crear DataLoader
dataset = TensorDataset(wheel_data, targets)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Entrenar el modelo
n_epochs = 100
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (wheel_data_batch, targets_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Calcular la salida del modelo
        outputs = model(wheel_data_batch)
        
        # Calcular la pérdida
        loss = criterion(outputs, targets_batch)
        
        # Retropropagar el error y actualizar los pesos del modelo
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

# Evaluar el modelo
model.eval()
with torch.no_grad():
    outputs = model(wheel_data)
    loss = criterion(outputs, targets)
    print('Loss: ', loss.item())

# Evaluar el modelo en todo el conjunto de datos
model.eval()
with torch.no_grad():
    outputs = model(wheel_data)

# Convertir los tensores de PyTorch a matrices NumPy
outputs = outputs.numpy()
targets = targets.numpy()

# Crear un scatter plot para la Dimensión 1
plt.figure(figsize=(8, 6))
plt.scatter(targets, targets, label='Objetivos reales (frecuencia)', c='blue')
plt.scatter(outputs, outputs, label='Predicciones (frecuencia)', c='red')
plt.xlabel('ground true')
plt.ylabel('prediccion')
plt.title('Gráfico de dispersión de frecuencia (Objetivos vs. Predicciones)')
plt.grid(True)
plt.legend()
plt.show()

# Seleccionar una fila de ejemplo del conjunto de datos de prueba
input_example = torch.tensor(X_test[0], dtype=torch.float32).view(1, 1, -1)

# Exportar el modelo a formato ONNX
torch.onnx.export(model, input_example, 'model_CNN_dinamica_1.onnx', input_names=['input'], output_names=['output'])