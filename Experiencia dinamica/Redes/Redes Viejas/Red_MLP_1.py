import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Cargar datos
file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/dataset_1_random_sin_nombre_exp_coma.csv'  # Reemplaza esto con la ruta de tu archivo CSV
data = pd.read_csv(file_path, delimiter=',')

# Extraer las columnas de entrada y salida
X = data[['Freq', 'PSD']].values
y = data[['frecuencia predominante', 'Tension resultante media']].values

# Normalizar los datos de entrada
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)  
        self.fc2 = nn.Linear(32, 2)   
        self.relu = nn.ReLU()         

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convertir los datos de entrenamiento a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
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
input_example = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0)

# Exportar el modelo a formato ONNX
torch.onnx.export(model, input_example, 'model_MLP_dinamica_1.onnx', input_names=['input'], output_names=['output'])
