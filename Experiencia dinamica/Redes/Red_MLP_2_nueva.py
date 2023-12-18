from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# Definir la arquitectura de la red
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Parámetros de la red
input_size = 756  # Número de características de entrada
hidden_size1 = 100  # Tamaño de la primera capa oculta
hidden_size2 = 50   # Tamaño de la segunda capa oculta
hidden_size3 = 20   # Tamaño de la tercera capa oculta
output_size = 4    # Número de características de salida

# Instanciar la red
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
num_epochs = 1  # Puedes ajustar esto
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluación
model.eval()
with torch.no_grad():
    predictions = []
    errors_by_feature = [[] for _ in range(output_size)]
    
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())
        
        batch_errors = ((outputs - targets)**2).mean(dim=0).sqrt().numpy()
        
        for i in range(output_size):
            errors_by_feature[i].append(batch_errors[i])

# Convertir las listas en arrays numpy
predictions = np.concatenate(predictions, axis=0)
y_test_np = y_test

# Gráfica del ground truth vs predicciones en subgráficos
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

for i, ax in enumerate(axes.flatten()):
    ax.scatter(y_test_np[:, i], predictions[:, i], label=f'Feature {i+1} (Predictions)', color='blue')
    ax.scatter(y_test_np[:, i], y_test_np[:, i], label=f'Feature {i+1} (Ground Truth)', color='red', marker='x')
    ax.set_title(f'Feature {i+1}')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')
    ax.legend()

plt.tight_layout()
plt.show()

# Calcular y mostrar el error medio por característica
for i in range(output_size):
    feature_error = np.mean(errors_by_feature[i])
    print(f'Average Error for Feature {i+1}: {feature_error:.4f}')
    
# El tamaño de entrada esperado es 'input_size'
input_size = 756  # Asegúrate de que esto coincida con la definición de tu modelo

# Crear un tensor de ejemplo con el tamaño de entrada correcto
# Este tensor debe ser de tipo float32 y tener la dimensión correcta
ejemplo_input = torch.randn(1, input_size, dtype=torch.float32)

# Exportar el modelo a ONNX
# Asegúrate de que los nombres de entrada y salida sean descriptivos y útiles
torch.onnx.export(model, ejemplo_input, 'model_MLP_DINAMICA_FINAL.onnx', 
                  input_names=['input'], output_names=['output'])
