import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
mat_fname = 'Datasets/mi_matriz_solo_diritletch_enriquesida.mat'
mat = sio.loadmat(mat_fname)
matriz_cargada = mat['dataset_matriz']

#primero mezclamos los casos
num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]
# Puedo hacer 138 casos de train y 58 de train (30%)
total_casos = matriz_cargada_mezclada.shape[0]
porcentaje_entrenamiento = 0.7
num_entrenamiento = int(total_casos * porcentaje_entrenamiento)
num_pruebas = total_casos - num_entrenamiento
# esto es canal 12 que contiene los bordes nomas. 
temp_train_dirichlet = matriz_cargada_mezclada[:num_entrenamiento, :, :, 12]
temp_test_dirichlet = matriz_cargada_mezclada[num_entrenamiento:, :, :, 12]
# esto es el ground truth. 
temp_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 17]
temp_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 17]
#convertis en tensores
temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()
temp_train_tensor_dirichlet = torch.from_numpy(temp_train_dirichlet).float()
temp_test_tensor_dirichlet = torch.from_numpy(temp_test_dirichlet).float()
# Añadir una dimensión extra para los canales 
temp_train_tensor = temp_train_tensor.unsqueeze(1)
temp_test_tensor = temp_test_tensor.unsqueeze(1)
temp_train_tensor_dirichlet = temp_train_tensor_dirichlet.unsqueeze(1)
temp_test_tensor_dirichlet = temp_test_tensor_dirichlet.unsqueeze(1)
# Crear TensorDatasets y DataLoaders
train_dataset = TensorDataset(temp_train_tensor, temp_train_tensor)
test_dataset = TensorDataset(temp_test_tensor, temp_test_tensor)
train_dataset_dirichlet = TensorDataset(temp_train_tensor_dirichlet, temp_train_tensor)
test_dataset_dirichlet = TensorDataset(temp_test_tensor_dirichlet, temp_test_tensor)
class HeatPropagationNet(nn.Module):
    def __init__(self):
        super(HeatPropagationNet, self).__init__()
        
        # Bloque de entrada
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.5)  # Añadir Leaky ReLU con una pendiente negativa de 0.1

        # Bloque intermedio
        self.conv2_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # Bloque de salida
        self.conv3_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # capa 1
        x1 = self.leaky_relu(self.conv1_1(x))  
        y1 = self.leaky_relu(self.conv1_2(x1))  
        
        # capa 2
        x2 = self.leaky_relu(self.conv1_1(y1))
        y2 = self.leaky_relu(self.conv1_2(x2))
        
        # capa 3
        x3 = self.leaky_relu(self.conv2_1(y2))
        y3 = self.leaky_relu(self.conv2_2(x3))
        
        # capa 4
        x4 = self.leaky_relu(self.conv2_1(y3))
        y4 = self.leaky_relu(self.conv2_2(x4))
        
        # capa 5
        x5 = self.leaky_relu(self.conv3_1(y4))
        y5 = self.leaky_relu(self.conv3_2(x5))
        
        # capa 6
        x6 = self.leaky_relu(self.conv3_1(y5))
        y6 = self.leaky_relu(self.conv3_2(x6))
        
        return y1, y2, y3, y4, y5, y6
    
def custom_loss(outputs, target):
    loss_tot = 0
    
    for output in outputs:  # Recorre todas las salidas de capa
        loss_tot+=F.mse_loss(output, target)

    return loss_tot

# Inicializa el escritor de TensorBoard
writer = SummaryWriter('runs/experiment_1')

# Verifica si CUDA está disponible y selecciona el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializar la red y el optimizador
model = HeatPropagationNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Define los pliegues
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)
# Inicializar la mejor pérdida a un número muy alto
best_val_loss = float('inf')
best_model_state = None

for fold, (train_index, val_index) in enumerate(kfold.split(train_dataset_dirichlet)):
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación
    train_subset = torch.utils.data.Subset(train_dataset_dirichlet, train_index)
    val_subset = torch.utils.data.Subset(train_dataset_dirichlet, val_index)

    # Crea los DataLoader para los conjuntos de entrenamiento y validación
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=32)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=32)

    # Aquí deberías inicializar y/o cargar tu modelo y optimizador
    model = HeatPropagationNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # Bucle de entrenamiento y validación
    for epoch in range(200):
        train_loss_total, train_loss_ultima, val_loss_total, val_loss_ultima = 0.0, 0.0, 0.0, 0.0 
        num_batches_train, num_batches_val = 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_all = outputs[:-1]
            output_final = outputs[-1:]
            loss_all = custom_loss(outputs_all, labels)
            loss_final = custom_loss(output_final, labels)
            loss_all.backward()
            optimizer.step()
            train_loss_total += loss_all.item()
            train_loss_ultima += loss_final.item()
            num_batches_train += 1
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs_all = outputs[:-1]
                output_final = outputs[-1:]
                loss_all_validation = custom_loss(outputs_all, labels)
                loss_ultima_validation = custom_loss(output_final, labels)
                val_loss_total += loss_all_validation.item()
                val_loss_ultima += loss_ultima_validation.item()
                num_batches_val += 1
        # Actualiza la mejor pérdida de validación y guarda el estado del modelo si es necesario
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_model_state = model.state_dict()
        train_loss_total /= num_batches_train
        train_loss_ultima /= num_batches_train
        val_loss_total /= num_batches_val
        val_loss_ultima /= num_batches_val
        
        if (epoch+1) % 50 == 0:
            print(f'Fold {fold}, Epoch {epoch+1}, Training Loss (total): {train_loss_total}, Validation Loss: {val_loss_total}')
            print(f'Fold {fold}, Epoch {epoch+1}, Training Loss (ultima capa): {train_loss_ultima}, Validation Loss: {val_loss_ultima}')
      

model = HeatPropagationNet().to(device)
model.load_state_dict(best_model_state)

model = HeatPropagationNet().to(device)
model.load_state_dict(best_model_state)

# Crear DataLoader para el conjunto de pruebas
test_dataset_dirichlet = TensorDataset(temp_test_tensor_dirichlet, temp_test_tensor)
test_loader = DataLoader(test_dataset_dirichlet, batch_size=32)

# Cambia el modelo a modo de evaluación
model.eval()

total_loss = 0
num_batches = 0

# No necesitamos calcular gradientes durante la evaluación
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Hacer una predicción con el modelo
        outputs = model(inputs)

        # Calcular la pérdida
        loss = custom_loss(outputs, labels)
        total_loss += loss.item()
        num_batches += 1

# Calcular la pérdida media
mean_loss = total_loss / num_batches

print('Mean Loss on Test Set:', mean_loss)

# Visualizar las salidas y ground truth una vez después de evaluar todo el conjunto de pruebas
# Crear una figura con una cuadrícula de subplots
fig, axs = plt.subplots(5, 2, figsize=(10, 20))

# Recorrer las 5 salidas y ground truth
for j in range(5):
    # Mostrar la salida de la red en el primer subplot de la fila j
    axs[j, 0].imshow(outputs[-1][j, 0].cpu().numpy(), cmap='hot')
    axs[j, 0].set_title("Output de la red")

    # Mostrar el ground truth en el segundo subplot de la fila j
    axs[j, 1].imshow(labels[j, 0].cpu().numpy(), cmap='hot')
    axs[j, 1].set_title("Ground Truth")

# Ajustar los espacios entre subplots y entre las figuras
fig.tight_layout()

# Mostrar la figura
plt.show()

# Crear una figura y un eje para la gráfica
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Obtener los valores del ground truth y la salida de la red
ground_truth = labels[:, 0].cpu().numpy().flatten()
red_output = outputs[-1][:, 0].cpu().numpy().flatten()

# Graficar los puntos del ground truth en rojo
plt.scatter(ground_truth, red_output, c='red', label='Ground Truth')

# Graficar los puntos de la salida de la red en azul
plt.scatter(red_output, red_output, c='blue', label='Salida de la red')

# Establecer etiquetas de los ejes
plt.xlabel("Salida de la red")
plt.ylabel("Ground Truth")

# Establecer título de la gráfica
plt.title("Dispersión de los puntos")

# Mostrar la leyenda
plt.legend()

# Mostrar la gráfica
plt.show()
from tensorboardX import SummaryWriter
# Asegúrate de que tanto el modelo como el tensor de entrada estén en el mismo dispositivo (por ejemplo, CPU)
device = torch.device("cpu")
model.to(device)
input_tensor = torch.randn(1, 1, 64, 64).to(device)

# Ahora, puedes agregar tu modelo al escritor de resumen
writer.add_graph(model, input_to_model=input_tensor)
