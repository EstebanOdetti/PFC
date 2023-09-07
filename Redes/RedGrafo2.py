import torch
from torch_geometric.data import Data
import numpy as np
import scipy.io as sio
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product

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

# Subseleccionar los canales 1, 2 y 15
selected_channels = [0, 1, 4, 5, 12, 17]  # indices empezando desde 0
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

# Añade estas características a los nodos en tu grafo
for i, (x, y) in enumerate(product(range(7), range(7))):
    G.nodes[i]['features'] = node_features[x, y]

# Usando los dos primeros canales como las coordenadas x, y
pos = {i: (G.nodes[i]['features'][0], G.nodes[i]['features'][1]) for i in G.nodes}

# Dibuja el grafo, con colores de nodo basados en el valor de una de las características
colors = [G.nodes[i]['features'][5] for i in G.nodes]  # Usa la quinta característica para el color (temperatura)

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
plt.imshow(imagen, cmap='hot')  # Utilizar cmap='hot' para representar temperaturas
plt.title(f'Caso 1')

plt.tight_layout()
plt.show()

# Asegurarse de usar los canales seleccionados
train_data_selected_tensor = torch.from_numpy(train_data_selected).float()
test_data_selected_tensor = torch.from_numpy(test_data_selected).float()

# Aplanar las matrices 7x7 a vectores 49x6 (porque ahora hay 6 canales)
train_x = train_data_selected_tensor.reshape(-1, 49, 6)
test_x = test_data_selected_tensor.reshape(-1, 49, 6)

# Obtén el canal 17 (índice 5 en selected_channels) de tus datos seleccionados
train_y = train_data_selected_tensor[:, :, :, 5].reshape(-1, 49)
test_y = test_data_selected_tensor[:, :, :, 5].reshape(-1, 49)

# Crear una lista de Data objects para entrenamiento y prueba
train_data_list = [Data(x=train_x[i], edge_index=edge_index, y=train_y[i]) for i in range(train_x.shape[0])]
test_data_list = [Data(x=test_x[i], edge_index=edge_index, y=test_y[i]) for i in range(test_x.shape[0])]

# Definir el modelo de red
class GraphNetwork(torch.nn.Module):
    def __init__(self):
        super(GraphNetwork, self).__init__()
        self.conv1 = GCNConv(6, 16)  # 6 características de entrada, 16 características de salida
        self.conv2 = GCNConv(16, 1)  # 16 características de entrada, 1 característica de salida (temperatura)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        return x.view(-1)

model = GraphNetwork()

# Parámetros
batch_size = 32
learning_rate = 0.01
num_epochs = 100

# DataLoaders
train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

# Función de pérdida y optimizador
criterion = torch.nn.MSELoss()  # Por ejemplo, pérdida cuadrática media para problemas de regresión
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import matplotlib.pyplot as plt

# Inicializa una lista para almacenar los valores de pérdida de cada época
loss_values = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    for batch in train_loader:
        # Extraer datos del lote
        batch_x, batch_edge_index, batch_y = batch.x, batch.edge_index, batch.y
        data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, batch_y)
        
        # Acumula el valor de la pérdida y cuenta los lotes
        epoch_loss += loss.item()
        batch_count += 1
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Almacena el promedio de la pérdida de la época
    epoch_loss /= batch_count
    loss_values.append(epoch_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Graficando los valores de pérdida
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.grid(True)
plt.show()

# Evaluación
model.eval()
with torch.no_grad():
    total_loss = 0
    predictions = [] # Almacenaremos todas las predicciones aquí
    for batch in test_loader:
        # Extraer datos del lote
        _, _, batch_y = batch.x, batch.edge_index, batch.y
        data = Data(x=batch.x, edge_index=batch.edge_index, y=batch.y)
        # Forward pass
        outputs = model(data)
        predictions.append(outputs)  # Guardar las predicciones de este lote
        loss = criterion(outputs, batch_y)
        total_loss += loss.item() * batch.num_graphs
    avg_loss = total_loss / len(test_data_list)
    print(f"Test Loss: {avg_loss:.4f}")

# Convertir la lista de tensores de predicciones en un único tensor
predictions_tensor = torch.cat(predictions).reshape(-1, 7, 7)  # Asumiendo que tus datos son 7x7

# Dibujar el ground truth y las predicciones para los primeros N casos de prueba
N = 10
for i in range(min(N, len(test_data_list))):
    plt.figure(figsize=(12, 5))
    
    # Dibujar el ground truth
    plt.subplot(1, 2, 1)
    ground_truth = test_y[i].reshape(7, 7).numpy()  # Convertir de tensor a numpy para plotear
    plt.imshow(ground_truth, cmap='hot')
    plt.title(f'Caso {i+1} - Ground Truth')
    plt.colorbar()
    
    # Dibujar la predicción
    plt.subplot(1, 2, 2)
    prediction = predictions_tensor[i].numpy()  # Convertir de tensor a numpy para plotear
    plt.imshow(prediction, cmap='hot')
    plt.title(f'Caso {i+1} - Predicción')
    plt.colorbar()

    plt.tight_layout()
    plt.show()