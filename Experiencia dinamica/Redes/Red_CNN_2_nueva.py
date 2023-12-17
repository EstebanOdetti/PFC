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


class CombinedCNN(nn.Module):
    def __init__(self, input_channels, num_rows, output_size):
        super(CombinedCNN, self).__init__()
        
        # Capas convolucionales de la rueda delantera
        self.front_wheel_conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.front_wheel_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.front_wheel_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.front_wheel_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Capas convolucionales de la rueda trasera
        self.rear_wheel_conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.rear_wheel_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.rear_wheel_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.rear_wheel_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Aplanamiento
        self.flatten = nn.Flatten()

        # Capa completamente conectada
        self.fc = nn.Linear(6016, output_size)  # Ajusta el tamaño de entrada basado en la salida de las capas convolucionales

    def forward(self, x):
        # Rueda delantera
        front_x = F.relu(self.front_wheel_conv1(x))
        front_x = self.front_wheel_pool1(front_x)
        front_x = F.relu(self.front_wheel_conv2(front_x))
        front_x = self.front_wheel_pool2(front_x)

        # Rueda trasera
        rear_x = F.relu(self.rear_wheel_conv1(x))
        rear_x = self.rear_wheel_pool1(rear_x)
        rear_x = F.relu(self.rear_wheel_conv2(rear_x))
        rear_x = self.rear_wheel_pool2(rear_x)

        # Combinar representaciones
        combined_x = torch.cat([front_x, rear_x], dim=2)  # Concatenar a lo largo de la dimensión de características

        # Aplanar y pasar por la capa completamente conectada
        combined_x = self.flatten(combined_x)
        output = self.fc(combined_x)

        return output

# Parámetros de la red CNN combinada
input_channels = 4  # Número de canales de entrada (correspondiente a las características)
num_rows = num_rows_per_case  # Número de filas por caso
output_size = 4  # Número de características de salida

# Parámetros de la red CNN combinada
input_channels = 4  # Número de canales de entrada (correspondiente a las características)
num_rows = num_rows_per_case  # Número de filas por caso
output_size = 4  # Número de características de salida

# Instanciar la red CNN combinada
combined_cnn_model = CombinedCNN(input_channels, num_rows, output_size)

# Definir la función de pérdida y el optimizador
criterion_combined_cnn = nn.MSELoss()
optimizer_combined_cnn = optim.Adam(combined_cnn_model.parameters(), lr=0.01)

# Convertir los datos a tensores de PyTorch (asegúrate de tener las dimensiones correctas)
X_train_tensor_combined_cnn = torch.tensor(X_train, dtype=torch.float32).view(-1, input_channels, num_rows)
y_train_tensor_combined_cnn = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor_combined_cnn = torch.tensor(X_test, dtype=torch.float32).view(-1, input_channels, num_rows)
y_test_tensor_combined_cnn = torch.tensor(y_test, dtype=torch.float32)

# Crear datasets y dataloaders para la CNN combinada
train_dataset_combined_cnn = TensorDataset(X_train_tensor_combined_cnn, y_train_tensor_combined_cnn)
test_dataset_combined_cnn = TensorDataset(X_test_tensor_combined_cnn, y_test_tensor_combined_cnn)

train_loader_combined_cnn = DataLoader(train_dataset_combined_cnn, batch_size=4, shuffle=True)
test_loader_combined_cnn = DataLoader(test_dataset_combined_cnn, batch_size=4, shuffle=False)

# Entrenamiento de la CNN combinada
num_epochs_combined_cnn = 150  # Puedes ajustar esto
for epoch in range(num_epochs_combined_cnn):
    for inputs, targets in train_loader_combined_cnn:
        optimizer_combined_cnn.zero_grad()
        outputs = combined_cnn_model(inputs)
        loss = criterion_combined_cnn(outputs, targets)
        loss.backward()
        optimizer_combined_cnn.step()
    print(f'Epoch [{epoch+1}/{num_epochs_combined_cnn}], Loss: {loss.item():.4f}')

# Evaluación de la CNN combinada
combined_cnn_model.eval()
with torch.no_grad():
    predictions_combined_cnn = []
    errors_by_feature_combined_cnn = [[] for _ in range(output_size)]
    
    for inputs, targets in test_loader_combined_cnn:
        outputs = combined_cnn_model(inputs)
        predictions_combined_cnn.append(outputs.numpy())
        
        batch_errors = ((outputs - targets)**2).mean(dim=0).sqrt().numpy()
        
        for i in range(output_size):
            errors_by_feature_combined_cnn[i].append(batch_errors[i])

# Convertir las listas en arrays numpy
predictions_combined_cnn = np.concatenate(predictions_combined_cnn, axis=0)



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