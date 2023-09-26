import torch
from torch_geometric.data import Data
import numpy as np
import scipy.io as sio
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ClusterGCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

# Cargar la matriz
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

selected_channels = [0, 1, 12, 17]  # indices empezando desde 0
train_data_selected = matriz_cargada_mezclada[:num_entrenamiento, :, :, selected_channels]
test_data_selected = matriz_cargada_mezclada[num_entrenamiento:, :, :, selected_channels]

# Crear objeto Data para PyTorch Geometric
# Primero necesitas crear edge_index, que define las conexiones del grafo.
# Crear un grafo 7x7 usando NetworkX
# Asumiendo que node_features tiene una forma de (num_samples, 7, 7, num_features)
node_features = train_data_selected[0]  # Solo toma el primer sample por ahora

# Crear un grafo 7x7 usando NetworkX
G = nx.grid_2d_graph(7, 7)

# Convertir nodos a índices enteros únicos
node_mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, node_mapping)

# Construir edge_index
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
print(edge_index)
# Añade estas características a los nodos en tu grafo
for i, (x, y) in enumerate(product(range(7), range(7))):
    G.nodes[i]['features'] = node_features[x, y]

# Usando los dos primeros canales como las coordenadas x, y
pos = {i: (G.nodes[i]['features'][0], G.nodes[i]['features'][1]) for i in G.nodes}

# Dibuja el grafo, con colores de nodo basados en el valor de una de las características
colors = [G.nodes[i]['features'][2] for i in G.nodes]  # Usa la quinta característica para el color (temperatura)

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
#mostras los 10 primeros casos
primer_caso = temp_train_tensor_dirichlet[0]
# Gráfico de NetworkX
plt.figure(figsize=(12, 6))

# Suponiendo que "colors" es una lista de los valores de temperatura de cada nodo
label_dict = {i: f'{val:.2f}' for i, val in enumerate(colors)}
plt.subplot(1, 2, 1)
nx.draw(G, pos, node_color=colors, cmap=plt.cm.Reds)
nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, verticalalignment='bottom')
plt.title('Graph Representation')

# Mostrar el primer caso
plt.subplot(1, 2, 2)
primer_caso = temp_train_tensor[0]
imagen = primer_caso[:, :]
plt.imshow(imagen, cmap='hot', origin='lower')  # Utilizar cmap='hot' para representar temperaturas
plt.title(f'Caso 1')

plt.tight_layout()
plt.show()

# Asegurarse de usar los canales seleccionados
train_data_selected_tensor = torch.from_numpy(train_data_selected).float()
test_data_selected_tensor = torch.from_numpy(test_data_selected).float()


# Función para transformar el tensor seleccionado en una lista de objetos Data
def tensor_to_data_list(data_tensor):
    data_list = []
    
    for sample in range(data_tensor.shape[0]):
        # Obtener características del nodo y valor objetivo para cada nodo
        x = data_tensor[sample, :, :, :-1].reshape(49, -1)  # Excluimos el último canal para x
        y = data_tensor[sample, :, :, -1].reshape(49)  # Usamos el último canal para y

        data = Data(x=x, edge_index=edge_index.clone(), y=y)
        data_list.append(data)
    
    return data_list

# Convertir tensores seleccionados en listas de objetos Data
train_data_list = tensor_to_data_list(train_data_selected_tensor)
test_data_list = tensor_to_data_list(test_data_selected_tensor)

# Definir el modelo de red
class GraphNetwork(torch.nn.Module):
    def __init__(self):
        super(GraphNetwork, self).__init__()
        
        # Entry block
        self.conv1_1 = ClusterGCNConv(3, 16)
        self.conv1_2 = ClusterGCNConv(16, 1)
        
        # Intermediate blocks
        self.conv2_1 = ClusterGCNConv(1, 16)
        self.conv2_2 = ClusterGCNConv(16, 1)
        
        self.conv3_1 = ClusterGCNConv(1, 16)
        self.conv3_2 = ClusterGCNConv(16, 1)
        
        # Exit block
        self.conv4_1 = ClusterGCNConv(1, 16)
        self.conv4_2 = ClusterGCNConv(16, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)  # Leaky ReLU with a negative slope of 0.2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Entry block
        x1 = self.leaky_relu(self.conv1_1(x, edge_index))
        y1 = self.leaky_relu(self.conv1_2(x1, edge_index))
        
        # Intermediate blocks
        x2 = self.leaky_relu(self.conv2_1(y1, edge_index))
        y2 = self.leaky_relu(self.conv2_2(x2, edge_index))
        
        x3 = self.leaky_relu(self.conv3_1(y2, edge_index))
        y3 = self.leaky_relu(self.conv3_2(x3, edge_index))
        
        # Exit block
        x4 = self.leaky_relu(self.conv4_1(y3, edge_index))
        y4 = self.leaky_relu(self.conv4_2(x4, edge_index))
        
        return y1.view(-1), y2.view(-1), y3.view(-1), y4.view(-1)

model = GraphNetwork()

# Verifica si CUDA está disponible y selecciona el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def custom_loss(outputs, target):
    loss_tot = 0
    
    for output in outputs:  # Recorre todas las salidas de capa
        loss_tot += F.mse_loss(output.squeeze(), target)

    return loss_tot

from sklearn.model_selection import KFold

# Parámetros
batch_size = 64
learning_rate = 0.0001
num_epochs = 500
k_folds = 2

# Usando KFold de sklearn
kfold = KFold(n_splits=k_folds, shuffle=True)

# Lista para almacenar pérdidas por cada fold
loss_values_kfold = []
# Inicializar la red y el optimizador para cada fold
model = model.to(device)  # Reemplaza con una nueva instancia de tu modelo si se modifica durante el entrenamiento
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data_list)):
    print(f"FOLD {fold+1}/{k_folds}")
    
    # Divide los datos en entrenamiento y validación para este fold
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    train_loader = DataLoader(train_data_list, batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(train_data_list, batch_size=batch_size, sampler=test_subsampler)
        
    # Inicializa una lista para almacenar los valores de pérdida de cada época
    loss_values = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for batch in train_loader:
            # Extraer datos del lote
            batch_x, batch_edge_index, batch_y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
            data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y).to(device)
            # Forward pass
            outputs = model(data)
            outputs_all = outputs[:-1]
            output_final = outputs[-1:]
            
            loss_final = F.mse_loss(output_final[-1].squeeze(), batch_y)
            # Calcular la pérdida utilizando tu función custom_loss
            loss_all = custom_loss(outputs_all, batch_y) + loss_final
            
            
            # Acumula el valor de la pérdida y cuenta los lotes
            epoch_loss += loss_all.item() + loss_final.item()
            batch_count += 1
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
        
        # Almacena el promedio de la pérdida de la época
        epoch_loss /= batch_count
        loss_values.append(epoch_loss)
        
        print(f"Epoca [{epoch+1}/{num_epochs}], Loss todas las capas: {epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss capa final: {loss_final:.4f}")
    
    loss_values_kfold.append(loss_values)


# Graficando los valores de pérdida
plt.plot(loss_values)
plt.xlabel('Epoca')
plt.ylabel('Perdida')
plt.title('Perdida por epoca')
plt.grid(True)
plt.show()

# Evaluar el modelo en el conjunto de prueba

# Preparar el DataLoader para el conjunto de prueba
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False) 

# Lista para almacenar las pérdidas del conjunto de prueba
test_losses = []

# Desactivar el cálculo de gradientes para mejorar la eficiencia
with torch.no_grad():
    model.eval()  # Cambiar el modelo a modo de evaluación

    total_loss = 0.0
    batch_count = 0
    
    for batch in test_loader:
        # Extraer datos del lote
        batch_x, batch_edge_index, batch_y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
        data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y).to(device)
        
        # Forward pass
        outputs = model(data)
        outputs_all = outputs[:-1]
        output_final = outputs[-1:]
        
        # Calcular la pérdida
        loss_all = custom_loss(outputs_all, batch_y)
        loss_final = F.mse_loss(output_final[-1].squeeze(), batch_y)
        
        # Acumula el valor de la pérdida y cuenta los lotes
        total_loss += loss_all.item() + loss_final.item()
        batch_count += 1

    # Calcula la pérdida promedio del conjunto de prueba
    average_test_loss = total_loss / batch_count
    test_losses.append(average_test_loss)

    print(f"Pérdida en el conjunto de prueba: {average_test_loss:.4f}")
    
# Evaluar el modelo en el conjunto de prueba

test_loader = DataLoader(test_data_list, batch_size=10, shuffle=True)  # Obtiene 10 muestras aleatorias

# Obtiene un lote de datos de prueba
batch = next(iter(test_loader))

# Extraer datos del lote
batch_x, batch_edge_index, batch_y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y).to(device)

# Forward pass para obtener las predicciones
with torch.no_grad():
    model.eval()
    outputs = model(data)
    predictions = outputs[-1].squeeze().cpu().numpy()
    ground_truth = batch_y.cpu().numpy()

# Graficar las predicciones y ground truth
plt.figure(figsize=(12, 6))
plt.plot(predictions, 'ro', label='Predicciones')
plt.plot(ground_truth, 'bo', label='Ground Truth')
plt.xlabel('Ejemplo')
plt.ylabel('Valor')
plt.title('Comparación de Predicciones vs Ground Truth')
plt.legend()
plt.grid(True)
plt.show()