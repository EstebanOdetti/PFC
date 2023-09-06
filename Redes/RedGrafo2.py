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
#mostras los 10 primeros casos
primeros_10_casos = temp_train_tensor_dirichlet[0:10]

# Subseleccionar los canales 1, 2 y 15
selected_channels = [0, 1, 4, 5, 12, 17]  # indices empezando desde 0
train_data_selected = matriz_cargada[:num_entrenamiento, :, :, selected_channels]
test_data_selected = matriz_cargada[num_entrenamiento:, :, :, selected_channels]

# Convertir a tensor PyTorch
train_data_tensor = torch.from_numpy(train_data_selected).float()
test_data_tensor = torch.from_numpy(test_data_selected).float()

# Ahora train_data_tensor y test_data_tensor tienen tamaño [num_casos, 7, 7, 3]

# Crear objeto Data para PyTorch Geometric
# Primero necesitas crear edge_index, que define las conexiones del grafo.
# Crear un grafo 7x7 usando NetworkX
G = nx.grid_2d_graph(7, 7)

# Convertir nodos a índices enteros únicos
node_mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, node_mapping)

# Construir edge_index
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

nx.draw(G, with_labels=True)
plt.show()
# Asegurarse de usar los canales seleccionados
train_data_selected_tensor = torch.from_numpy(train_data_selected).float()
test_data_selected_tensor = torch.from_numpy(test_data_selected).float()

# Aplanar las matrices 7x7 a vectores 49x6 (porque ahora hay 6 canales)
train_x = train_data_selected_tensor.reshape(-1, 49, 6)
test_x = test_data_selected_tensor.reshape(-1, 49, 6)

# El objetivo y, en este caso, sería la temperatura (tomada del canal que quieras, por ejemplo, el 17)
train_y = temp_train_tensor.reshape(-1, 49)
test_y = temp_test_tensor.reshape(-1, 49)

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
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x.view(-1)


model = GraphNetwork()

# Parámetros
batch_size = 32
learning_rate = 0.001
num_epochs = 100

# DataLoaders
train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

# Función de pérdida y optimizador
criterion = torch.nn.MSELoss()  # Por ejemplo, pérdida cuadrática media para problemas de regresión
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Extraer datos del lote
        _, _, batch_y = batch.x, batch.edge_index, batch.y
        data = Data(x=batch.x, edge_index=batch.edge_index, y=batch.y)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, batch_y)
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluación
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch in test_loader:
        # Extraer datos del lote
        _, _, batch_y = batch.x, batch.edge_index, batch.y
        data = Data(x=batch.x, edge_index=batch.edge_index, y=batch.y)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item() * batch.num_graphs
    avg_loss = total_loss / len(test_data_list)
    print(f"Test Loss: {avg_loss:.4f}")