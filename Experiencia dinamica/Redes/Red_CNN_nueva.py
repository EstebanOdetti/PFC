from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

#Directorio base donde se encuentran tus archivos
directorio_base = os.path.dirname(__file__)  # Obtener el directorio del script actual

# Leer y preparar el conjunto de datos
combined_data_path = os.path.join(directorio_base, 'Datasets\FINAL', 'combined_data_2columnas.csv')
simulations_data_path = os.path.join(directorio_base, 'Datasets\FINAL', 'Mediciones_simulaciones.csv')

combined_data = pd.read_csv(combined_data_path)
mediciones_simulaciones = pd.read_csv(simulations_data_path)

# Reestructurar los datos: cada 189 filas representan un caso
num_rows_per_case = 189
num_cases = len(combined_data) // num_rows_per_case
X = combined_data.iloc[:, [0, 1, 3, 4]].values.reshape(num_cases, num_rows_per_case * 4)
y = mediciones_simulaciones.iloc[:, 1:5].values

# Comprobar si las dimensiones de X y y son compatibles
if X.shape[0] != y.shape[0]:
    raise ValueError("La cantidad de casos en 'combined_data' y 'Mediciones_simulaciones' no coincide.")

# Dividir los datos en conjuntos de entrenamiento y prueba, manteniendo el orden
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Mostrar un ejemplo de un caso
example_case = X[0, :].reshape(num_rows_per_case, 4)
example_case_df = pd.DataFrame(example_case, columns=['Freq_adel', 'PSD_adel', 'Freq_atr', 'PSD_atr'])

# Convertir los datos a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Crear datasets y dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Definir la arquitectura de la red CNN
class CNN(nn.Module):
    def __init__(self, input_channels, num_rows, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * num_rows, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# Parámetros de la red CNN
input_channels = 4  # Número de canales de entrada (correspondiente a las características)
num_rows = num_rows_per_case  # Número de filas por caso
output_size = 4    # Número de características de salida

# Instanciar la red CNN
cnn_model = CNN(input_channels, num_rows, output_size)

# Definir la función de pérdida y el optimizador
criterion_cnn = nn.MSELoss()
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.01)

# Convertir los datos a tensores de PyTorch (asegúrate de tener las dimensiones correctas)
X_train_tensor_cnn = torch.tensor(X_train, dtype=torch.float32).view(-1, input_channels, num_rows)
y_train_tensor_cnn = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor_cnn = torch.tensor(X_test, dtype=torch.float32).view(-1, input_channels, num_rows)
y_test_tensor_cnn = torch.tensor(y_test, dtype=torch.float32)

# Crear datasets y dataloaders para la CNN
train_dataset_cnn = TensorDataset(X_train_tensor_cnn, y_train_tensor_cnn)
test_dataset_cnn = TensorDataset(X_test_tensor_cnn, y_test_tensor_cnn)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=4, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=4, shuffle=False)

# Entrenamiento de la CNN
num_epochs_cnn = 150  # Puedes ajustar esto
for epoch in range(num_epochs_cnn):
    for inputs, targets in train_loader_cnn:
        optimizer_cnn.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion_cnn(outputs, targets)
        loss.backward()
        optimizer_cnn.step()
    print(f'Epoch [{epoch+1}/{num_epochs_cnn}], Loss: {loss.item():.4f}')

# Evaluación de la CNN
cnn_model.eval()
with torch.no_grad():
    predictions_cnn = []
    errors_by_feature_cnn = [[] for _ in range(output_size)]
    
    for inputs, targets in test_loader_cnn:
        outputs = cnn_model(inputs)
        predictions_cnn.append(outputs.numpy())
        
        batch_errors = ((outputs - targets)**2).mean(dim=0).sqrt().numpy()
        
        for i in range(output_size):
            errors_by_feature_cnn[i].append(batch_errors[i])

# Convertir las listas en arrays numpy
predictions_cnn = np.concatenate(predictions_cnn, axis=0)

# Reformatear predictions_cnn a 2D si es necesario
predictions_cnn = predictions_cnn.reshape(-1, output_size)

y_test_np = y_test

# Gráfica del ground truth vs predicciones en subgráficos
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

for i, ax in enumerate(axes.flatten()):
    ax.scatter(y_test_np[:, i], predictions_cnn[:, i], label=f'Feature {i+1} (Predictions)', color='blue')  # Usar predictions_cnn en lugar de predictions
    ax.scatter(y_test_np[:, i], y_test_np[:, i], label=f'Feature {i+1} (Ground Truth)', color='red', marker='x')
    ax.set_title(f'Feature {i+1}')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')
    ax.legend()

plt.tight_layout()
plt.show()

# Calcular y mostrar el error medio por característica
for i in range(output_size):
    feature_error = np.mean(errors_by_feature_cnn[i])
    print(f'Average Error for Feature {i+1}: {feature_error:.4f}')