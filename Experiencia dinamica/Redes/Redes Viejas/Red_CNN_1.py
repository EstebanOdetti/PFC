import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Cargar datos
file_path_entradas = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/FINAL/combined_data.csv' 
data_entradas = pd.read_csv(file_path_entradas, delimiter=',')
file_path_salidas = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/FINAL/Mediciones_simulaciones.csv'  
data_salidas = pd.read_csv(file_path_salidas, delimiter=',')
# Extraer las columnas de entrada y salida
X = data_entradas[['Freq', 'PSD']].values
y = data_salidas[['P1_Frecuencia', 'P1_RMS', 'P2_Frecuencia', 'P2_RMS']].values

# Normalizar los datos de entrada
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 2, padding=1)
        self.conv2 = nn.Conv1d(64, 32, 2, padding=1)
        self.fc1 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# Convertir los datos de entrenamiento a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 2)  # Cambiar la forma del tensor
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


# Instanciar el modelo, la función de pérdida y el optimizador
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar la red neuronal
epochs = 10
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
        
# Seleccionar una fila de ejemplo del conjunto de datos de prueba
input_example = torch.tensor(X_test[0], dtype=torch.float32).view(1, 1, -1)

# Exportar el modelo a formato ONNX
torch.onnx.export(model, input_example, 'model_CNN_dinamica_1.onnx', input_names=['input'], output_names=['output'])