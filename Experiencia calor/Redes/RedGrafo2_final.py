import torch
from torch_geometric.data import Data
import numpy as np
import scipy.io as sio
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GraphConv, ResGatedGraphConv, TAGConv, ARMAConv, ClusterGCNConv, GeneralConv, HGTConv
from torch_geometric.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
import torch.nn as nn
from sklearn.model_selection import KFold


mat_fname = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia calor/Datasets/mi_matriz_solo_diritletch_enriquesida.mat'
mat = sio.loadmat(mat_fname)
matriz_cargada = mat['dataset_matriz']


num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]

total_casos = matriz_cargada_mezclada.shape[0]
porcentaje_entrenamiento = 0.7
num_entrenamiento = int(total_casos * porcentaje_entrenamiento)


selected_channels = [0, 1, 12, 17]
train_data_selected = matriz_cargada_mezclada[:num_entrenamiento, :, :, selected_channels]
test_data_selected = matriz_cargada_mezclada[num_entrenamiento:, :, :, selected_channels]





node_features = train_data_selected[0]


G = nx.grid_2d_graph(7, 7)


node_mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, node_mapping)


edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()


for i, (x, y) in enumerate(product(range(7), range(7))):
    G.nodes[i]['features'] = node_features[x, y]


pos = {i: (G.nodes[i]['features'][0], G.nodes[i]['features'][1]) for i in G.nodes}


colors = [G.nodes[i]['features'][3] for i in G.nodes]

temp_train_dirichlet = matriz_cargada_mezclada[:num_entrenamiento, :, :, 12]
temp_test_dirichlet = matriz_cargada_mezclada[num_entrenamiento:, :, :, 12]

temp_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 17]
temp_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 17]

temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()
temp_train_tensor_dirichlet = torch.from_numpy(temp_train_dirichlet).float()
temp_test_tensor_dirichlet = torch.from_numpy(temp_test_dirichlet).float()

primer_caso = temp_train_tensor_dirichlet[0]


vmin = min(colors)
vmax = max(colors)

plt.figure(figsize=(12, 6))


label_dict = {i: f'{val:.2f}' for i, val in enumerate(colors)}

plt.subplot(1, 2, 1)

nx.draw(G, pos, node_color=colors, cmap=plt.cm.Reds, vmin=vmin, vmax=vmax)
nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, verticalalignment='bottom')
plt.title('Grafo de temperatura')

plt.subplot(1, 2, 2)
primer_caso = temp_train_tensor[0]
imagen = primer_caso[:, :]

plt.imshow(imagen, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
plt.title(f'Caso 1 del dataset')

plt.tight_layout()
plt.colorbar()
plt.show()


train_data_selected_tensor = torch.from_numpy(train_data_selected).float()
test_data_selected_tensor = torch.from_numpy(test_data_selected).float()


train_x = train_data_selected_tensor[:, :, :, 0:3].reshape(-1, 49, 3)
test_x = test_data_selected_tensor[:, :, :, 0:3].reshape(-1, 49, 3)


train_y = train_data_selected_tensor[:, :, :, 3].reshape(-1, 49)
test_y = test_data_selected_tensor[:, :, :, 3].reshape(-1, 49)


train_data_list = [Data(x=train_x[i], edge_index=edge_index, y=train_y[i]) for i in range(train_x.shape[0])]
test_data_list = [Data(x=test_x[i], edge_index=edge_index, y=test_y[i]) for i in range(test_x.shape[0])]


class GraphNetwork(torch.nn.Module):
    def __init__(self):
        super(GraphNetwork, self).__init__()
        self.conv1 = ClusterGCNConv(3, 16)
        self.conv2 = ClusterGCNConv(16, 16)
        self.conv3 = ClusterGCNConv(16, 16)
        self.conv4 = ClusterGCNConv(16, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, 0.1)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, 0.1)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x, 0.1)
        return x.view(-1)

model = GraphNetwork()


batch_size = 64
learning_rate = 0.01
num_epochs = 200


train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)


criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)


val_loss_values = []
train_loss_values = []
for fold, (train_indices, val_indices) in enumerate(kfold.split(train_data_list)):

    train_data = [train_data_list[i] for i in train_indices]
    val_data = [train_data_list[i] for i in val_indices]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


    model = GraphNetwork()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for batch in train_loader:
            batch_x, batch_edge_index, batch_y = batch.x, batch.edge_index, batch.y
            outputs = model(batch_x, batch_edge_index)
            loss = criterion(outputs, batch_y)
            epoch_loss += loss.item()
            batch_count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= batch_count
        train_loss_values.append(epoch_loss)

        model.eval()
        val_epoch_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_edge_index, batch_y = batch.x, batch.edge_index, batch.y
                outputs = model(batch_x, batch_edge_index)
                loss = criterion(outputs, batch_y)
                val_epoch_loss += loss.item()
                val_batch_count += 1
        
        val_epoch_loss /= val_batch_count
        val_loss_values.append(val_epoch_loss)
        
        print(f"Fold [{fold+1}/5], Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")



plt.plot(val_loss_values)
plt.xlabel('Epoca')
plt.ylabel('Perdida de Validación')
plt.title('Perdida de Validación por Epoca')
plt.grid(True)
plt.show()


plt.plot(train_loss_values)
plt.xlabel('Epoca')
plt.ylabel('Perdida de entrenamiento')
plt.title('Perdida de entrenamiento por Epoca')
plt.grid(True)
plt.show()


model.eval()
with torch.no_grad():
    total_loss = 0
    predictions = []
    for batch in test_loader:

        batch_x, batch_edge_index, batch_y = batch.x, batch.edge_index, batch.y        

        outputs = model(batch_x, batch_edge_index)
        predictions.append(outputs)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item() * batch.num_graphs
    avg_loss = total_loss / len(test_data_list)
    print(f"Test Loss: {avg_loss:.4f}")


predictions_tensor = torch.cat(predictions).reshape(-1, 7, 7)

model = model.to('cpu')
batch_x = batch_x.to('cpu')
batch_edge_index = batch_edge_index.to('cpu')

N = 2

plt.figure(figsize=(26, 5*N)) 

for i in range(min(N, len(test_data_list))):

    plt.subplot(N, 2, 2*i + 1)
    ground_truth = test_y[i].reshape(7, 7).cpu().numpy()
    plt.imshow(ground_truth, cmap='hot')
    plt.title(f'Caso {i+1} - Ground Truth', fontsize=10)
    plt.colorbar()
    

    plt.subplot(N, 2, 2*i + 2)
    prediction = predictions_tensor[i].cpu().numpy()
    plt.imshow(prediction, cmap='hot')
    plt.title(f'Caso {i+1} - Predicción', fontsize=10)
    plt.colorbar()

plt.tight_layout()
plt.show()


predictions_flat = predictions_tensor.view(-1).cpu().numpy()
test_y_flat = torch.cat([batch.y for batch in test_loader]).view(-1).cpu().numpy()


plt.figure(figsize=(10, 10))

plt.scatter(test_y_flat, test_y_flat, c='red', label='Ground Truth', marker='x')

plt.scatter(test_y_flat, predictions_flat, c='blue', label='Salida de la red')


plt.xlabel("Ground Truth")
plt.ylabel("Salida de la red")


plt.title("Dispersión de los puntos")


plt.legend()


plt.show()


input_tensor, _ = test_loader.dataset[0]
input_example = input_tensor.unsqueeze(0).to('cpu')


torch.onnx.export(model, input_example, 'model_grafo_mejorada.onnx', input_names=['input'], output_names=['output'])