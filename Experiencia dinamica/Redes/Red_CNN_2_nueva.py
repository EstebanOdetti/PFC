import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Directorio base donde se encuentran tus archivos
directorio_base = os.path.dirname(__file__)  # Obtener el directorio del script actual

# Leer y preparar el conjunto de datos
matriz_total_data_path = os.path.join(directorio_base, 'Datasets', 'matriz_total.csv')
simulations_data_path = os.path.join(directorio_base, 'Datasets\FINAL', 'Mediciones_simulaciones.csv')

# Cargar datos
matriz_total = pd.read_csv(matriz_total_data_path)
mediciones_simulaciones = pd.read_csv(simulations_data_path)

# Asegurar que las dimensiones son correctas
assert matriz_total.shape[0] == 3654

# Número de filas que representa un caso y número total de casos
num_rows_per_case = 63
num_cases = 29

# Dividir matriz_total en dos: adelante (rueda delantera) y atrás (rueda trasera)
half_index = len(matriz_total) // 2
matriz_adelante = matriz_total.iloc[:half_index, :4]  # Seleccionar solo las primeras 4 columnas
matriz_atras = matriz_total.iloc[half_index:, :4]    # Seleccionar solo las primeras 4 columnas

# Redimensionar los datos para que tengan la forma (num_cases, 4, num_rows_per_case, 1)
X_adelante = matriz_adelante.values.reshape(num_cases, 4, num_rows_per_case, 1)
X_atras = matriz_atras.values.reshape(num_cases, 4, num_rows_per_case, 1)

# Preparar las etiquetas (salidas) correspondientes a cada caso
y = mediciones_simulaciones[['P1_Frecuencia', 'P1_RMS', 'P2_Frecuencia', 'P2_RMS']].values

# Dividir en conjuntos de entrenamiento y prueba
X_adelante_train, X_adelante_test, y_train, y_test = train_test_split(X_adelante, y, test_size=0.2, random_state=0, shuffle=False)
X_atras_train, X_atras_test, _, _ = train_test_split(X_atras, y, test_size=0.2, random_state=0, shuffle=False)

# Puedes imprimir las formas para confirmar
print(X_adelante_train.shape, X_adelante_test.shape, X_atras_train.shape, X_atras_test.shape, y_train.shape, y_test.shape)

class DualCNN(nn.Module):
    def __init__(self):
        super(DualCNN, self).__init__()
        # Rama para mediciones delanteras
        self.conv1_delante = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_delante = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        #self.conv3_delante = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv4_delante = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Rama para mediciones traseras
        self.conv1_atras = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_atras = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        #self.conv3_atras = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv4_atras = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Definición de la CNN de Fusión
        self.fusion_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fusion_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Tamaño de la salida de las capas convolucionales de fusión
        # Ajustar según las dimensiones reales
        salida_fusion_size = 256 * num_rows_per_case * 1
        print(salida_fusion_size)
        # Capa completamente conectada
        self.fc = nn.Linear(salida_fusion_size, 4)

    def forward(self, x_delante, x_atras):
        # Rama delantera
        x1 = F.relu(self.conv1_delante(x_delante))
        x1 = F.relu(self.conv2_delante(x1))

        # Rama trasera
        x2 = F.relu(self.conv1_atras(x_atras))
        x2 = F.relu(self.conv2_atras(x2))

        # Concatenar manteniendo las dimensiones espaciales
        x = torch.cat((x1, x2), dim=1)

        # Pasar por la CNN de Fusión
        x = F.relu(self.fusion_conv1(x))
        x = F.relu(self.fusion_conv2(x))

        # Aplanar antes de la capa completamente conectada
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Convertir los datos de NumPy a tensores de PyTorch
X_adelante_train_tensor = torch.tensor(X_adelante_train, dtype=torch.float32)
X_atras_train_tensor = torch.tensor(X_atras_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_adelante_test_tensor = torch.tensor(X_adelante_test, dtype=torch.float32)
X_atras_test_tensor = torch.tensor(X_atras_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Instanciar el modelo
model = DualCNN()

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Entrenamiento
num_epochs = 1  # Ajusta según sea necesario

for epoch in range(num_epochs):
    model.train()
    for i in range(X_adelante_train_tensor.size(0)):
        # Obtener lotes de datos
        data_delante = X_adelante_train_tensor[i].unsqueeze(0)
        data_atras = X_atras_train_tensor[i].unsqueeze(0)
        labels = y_train_tensor[i].unsqueeze(0)

        # Paso hacia adelante
        outputs = model(data_delante, data_atras)
        loss = criterion(outputs, labels)

        # Paso hacia atrás y optimizar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

model.eval()
predictions = []
with torch.no_grad():
    for i in range(len(X_adelante_test)):
        data_delante = torch.tensor(X_adelante_test[i]).unsqueeze(0).float()
        data_atras = torch.tensor(X_atras_test[i]).unsqueeze(0).float()
        output = model(data_delante, data_atras)
        predictions.append(output.numpy())

predictions = np.vstack(predictions)

# Suponiendo que 'predictions' es un array con tus predicciones y 'y_test' son los valores reales
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
features = ['P1_Frecuencia', 'P1_RMS', 'P2_Frecuencia', 'P2_RMS']
mse_per_feature = {}

for i, feature in enumerate(features):
    mse = mean_squared_error(y_test[:, i], predictions[:, i])
    mse_per_feature[feature] = mse

# Mostrar el MSE por cada característica
for feature, mse in mse_per_feature.items():
    print(f"MSE para {feature}: {mse}")
    
# Graficar el ground truth vs predicciones
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ['P1_Frecuencia', 'P1_RMS', 'P2_Frecuencia', 'P2_RMS']

for i, feature in enumerate(features):
    ax = axes[i//2, i%2]
    # Ground truth en puntos
    ax.scatter(range(len(y_test[:, i])), y_test[:, i], alpha=0.7, label='Ground Truth')
    # Predicciones en puntos
    ax.scatter(range(len(predictions[:, i])), predictions[:, i], alpha=0.7, label='Predictions')
    ax.set_xlabel('Samples')
    ax.set_ylabel(feature)
    ax.set_title(f'Comparison for {feature}')
    ax.legend()

plt.tight_layout()
plt.show()

# Crear un tensor de ejemplo para 'x_delante' y 'x_atras'
# Estos tensores deben ser de tipo float32 y tener las dimensiones correctas
ejemplo_x_delante = torch.randn(1, 4, num_rows_per_case, 1, dtype=torch.float32)
ejemplo_x_atras = torch.randn(1, 4, num_rows_per_case, 1, dtype=torch.float32)

# Exportar el modelo a ONNX
# Asegúrate de que los nombres de entrada y salida sean descriptivos y útiles
torch.onnx.export(model, (ejemplo_x_delante, ejemplo_x_atras), 'model_CNN_EXPDINAMICA_mejorada.onnx',
                  input_names=['x_delante_input', 'x_atras_input'], output_names=['output'])
