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


mat_fname = "Datasets/mi_matriz_solo_diritletch_enriquesida.mat"
mat = sio.loadmat(mat_fname)
matriz_cargada = mat["dataset_matriz"]


num_casos, _, _, _ = matriz_cargada.shape
indices_aleatorios = np.random.permutation(num_casos)
matriz_cargada_mezclada = matriz_cargada[indices_aleatorios]

total_casos = matriz_cargada_mezclada.shape[0]
porcentaje_entrenamiento = 0.7
num_entrenamiento = int(total_casos * porcentaje_entrenamiento)

selected_channels = [0, 1, 12, 17]
train_data_selected = matriz_cargada_mezclada[
    :num_entrenamiento, :, :, selected_channels
]
test_data_selected = matriz_cargada_mezclada[
    num_entrenamiento:, :, :, selected_channels
]


node_features = train_data_selected[0]


G = nx.grid_2d_graph(7, 7)


node_mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, node_mapping)


edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
print(edge_index)

for i, (x, y) in enumerate(product(range(7), range(7))):
    G.nodes[i]["features"] = node_features[x, y]


pos = {i: (G.nodes[i]["features"][0], G.nodes[i]["features"][1]) for i in G.nodes}


colors = [G.nodes[i]["features"][2] for i in G.nodes]

temp_train_dirichlet = matriz_cargada_mezclada[:num_entrenamiento, :, :, 12]
temp_test_dirichlet = matriz_cargada_mezclada[num_entrenamiento:, :, :, 12]

temp_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 17]
temp_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 17]

temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()
temp_train_tensor_dirichlet = torch.from_numpy(temp_train_dirichlet).float()
temp_test_tensor_dirichlet = torch.from_numpy(temp_test_dirichlet).float()

primer_caso = temp_train_tensor_dirichlet[0]

plt.figure(figsize=(12, 6))


label_dict = {i: f"{val:.2f}" for i, val in enumerate(colors)}
plt.subplot(1, 2, 1)
nx.draw(G, pos, node_color=colors, cmap=plt.cm.Reds)
nx.draw_networkx_labels(
    G, pos, labels=label_dict, font_size=8, verticalalignment="bottom"
)
plt.title("Graph Representation")


plt.subplot(1, 2, 2)
primer_caso = temp_train_tensor[0]
imagen = primer_caso[:, :]
plt.imshow(imagen, cmap="hot", origin="lower")
plt.title(f"Caso 1")

plt.tight_layout()
plt.show()


train_data_selected_tensor = torch.from_numpy(train_data_selected).float()
test_data_selected_tensor = torch.from_numpy(test_data_selected).float()


def tensor_to_data_list(data_tensor):
    data_list = []

    for sample in range(data_tensor.shape[0]):

        x = data_tensor[sample, :, :, :-1].reshape(49, -1)
        y = data_tensor[sample, :, :, -1].reshape(49)

        data = Data(x=x, edge_index=edge_index.clone(), y=y)
        data_list.append(data)

    return data_list


train_data_list = tensor_to_data_list(train_data_selected_tensor)
test_data_list = tensor_to_data_list(test_data_selected_tensor)


class GraphNetwork(torch.nn.Module):
    def __init__(self):
        super(GraphNetwork, self).__init__()

        self.conv1_1 = ClusterGCNConv(3, 16)
        self.conv1_2 = ClusterGCNConv(16, 1)

        self.conv2_1 = ClusterGCNConv(1, 16)
        self.conv2_2 = ClusterGCNConv(16, 1)

        self.conv3_1 = ClusterGCNConv(1, 16)
        self.conv3_2 = ClusterGCNConv(16, 1)

        self.conv4_1 = ClusterGCNConv(1, 16)
        self.conv4_2 = ClusterGCNConv(16, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.leaky_relu(self.conv1_1(x, edge_index))
        y1 = self.leaky_relu(self.conv1_2(x1, edge_index))

        x2 = self.leaky_relu(self.conv2_1(y1, edge_index))
        y2 = self.leaky_relu(self.conv2_2(x2, edge_index))

        x3 = self.leaky_relu(self.conv3_1(y2, edge_index))
        y3 = self.leaky_relu(self.conv3_2(x3, edge_index))

        x4 = self.leaky_relu(self.conv4_1(y3, edge_index))
        y4 = self.leaky_relu(self.conv4_2(x4, edge_index))

        return y1.view(-1), y2.view(-1), y3.view(-1), y4.view(-1)


model = GraphNetwork()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def custom_loss(outputs, target):
    loss_tot = 0

    for output in outputs:
        loss_tot += F.mse_loss(output.squeeze(), target)

    return loss_tot


from sklearn.model_selection import KFold


batch_size = 64
learning_rate = 0.0001
num_epochs = 500
k_folds = 2


kfold = KFold(n_splits=k_folds, shuffle=True)


loss_values_kfold = []

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data_list)):
    print(f"FOLD {fold+1}/{k_folds}")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = DataLoader(
        train_data_list, batch_size=batch_size, sampler=train_subsampler
    )
    test_loader = DataLoader(
        train_data_list, batch_size=batch_size, sampler=test_subsampler
    )

    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for batch in train_loader:

            batch_x, batch_edge_index, batch_y = (
                batch.x.to(device),
                batch.edge_index.to(device),
                batch.y.to(device),
            )
            data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y).to(device)

            outputs = model(data)
            outputs_all = outputs[:-1]
            output_final = outputs[-1:]

            loss_final = F.mse_loss(output_final[-1].squeeze(), batch_y)

            loss_all = custom_loss(outputs_all, batch_y) + loss_final

            epoch_loss += loss_all.item() + loss_final.item()
            batch_count += 1

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        epoch_loss /= batch_count
        loss_values.append(epoch_loss)

        print(f"Epoca [{epoch+1}/{num_epochs}], Loss todas las capas: {epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss capa final: {loss_final:.4f}")

    loss_values_kfold.append(loss_values)


plt.plot(loss_values)
plt.xlabel("Epoca")
plt.ylabel("Perdida")
plt.title("Perdida por epoca")
plt.grid(True)
plt.show()


test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)


test_losses = []


with torch.no_grad():
    model.eval()

    total_loss = 0.0
    batch_count = 0

    for batch in test_loader:

        batch_x, batch_edge_index, batch_y = (
            batch.x.to(device),
            batch.edge_index.to(device),
            batch.y.to(device),
        )
        data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y).to(device)

        outputs = model(data)
        outputs_all = outputs[:-1]
        output_final = outputs[-1:]

        loss_all = custom_loss(outputs_all, batch_y)
        loss_final = F.mse_loss(output_final[-1].squeeze(), batch_y)

        total_loss += loss_all.item() + loss_final.item()
        batch_count += 1

    average_test_loss = total_loss / batch_count
    test_losses.append(average_test_loss)

    print(f"Pérdida en el conjunto de prueba: {average_test_loss:.4f}")


test_loader = DataLoader(test_data_list, batch_size=10, shuffle=True)


batch = next(iter(test_loader))


batch_x, batch_edge_index, batch_y = (
    batch.x.to(device),
    batch.edge_index.to(device),
    batch.y.to(device),
)
data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y).to(device)


with torch.no_grad():
    model.eval()
    outputs = model(data)
    predictions = outputs[-1].squeeze().cpu().numpy()
    ground_truth = batch_y.cpu().numpy()


plt.figure(figsize=(12, 6))
plt.plot(predictions, "ro", label="Predicciones")
plt.plot(ground_truth, "bo", label="Ground Truth")
plt.xlabel("Ejemplo")
plt.ylabel("Valor")
plt.title("Comparación de Predicciones vs Ground Truth")
plt.legend()
plt.grid(True)
plt.show()
