import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

# Crear un grafo 7x7 usando NetworkX
G = nx.grid_2d_graph(7, 7)

# Convertir nodos a índices enteros únicos
node_mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, node_mapping)

# Construir edge_index
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

# Crear características aleatorias para los nodos
random_features = torch.randn(49, 16)  # 49 nodos, 16 características por nodo

# Añadir coordenadas espaciales como características
coordinates = torch.tensor([(i // 7, i % 7) for i in range(49)], dtype=torch.float)
x = torch.cat([random_features, coordinates], dim=1)  # 49 nodos, 18 características por nodo

# Crear etiquetas aleatorias para los nodos
y = torch.randint(0, 2, (49,))  # 49 nodos, etiquetas 0 o 1

# Crear objeto Data de PyTorch Geometric
data = Data(x=x, edge_index=edge_index, y=y)

# Dibujar el grafo
def plot_graph(G, node_color='skyblue'):
    pos = {node: (node % 7, node // 7) for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color=node_color, font_weight='bold', node_size=700, font_size=18)
    plt.show()

# Convertir el grafo de PyTorch Geometric a NetworkX para la visualización
G_nx = nx.Graph()
G_nx.add_edges_from(edge_index.t().numpy())
plot_graph(G_nx)

# Crear objeto Data de PyTorch Geometric
data = Data(x=x, edge_index=edge_index, y=y)

# Definir el modelo de la Red Neuronal de Grafo
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(18, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Entrenar el modelo
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluar el modelo
model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred.eq(data.y).sum().item())
accuracy = correct / data.y.size(0)
print(f'Accuracy: {accuracy}')