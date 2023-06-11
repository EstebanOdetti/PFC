from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch

class HeatPropagationNet(nn.Module):
    def __init__(self):
        super(HeatPropagationNet, self).__init__()
        # Bloque de entrada
        self.conv1_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)

        # Bloque intermedio
        self.conv2_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)

        # Bloque de salida
        self.conv3_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        #self.dropout = nn.Dropout2d(p=0.2)  # Añade una capa de Dropout. 'p' es la probabilidad de que cada nodo se apague.

    def forward(self, x):
        #capa1
        x1 = F.relu(self.conv1_1(x))
        #y1 = self.dropout(self.conv1_2(x1))
        y1 = self.conv1_2(x1)
        #capa2
        x2 = F.relu(self.conv2_1(y1))
        y2 = self.conv2_2(x2)
        #capa3
        x3 = F.relu(self.conv3_1(y2))
        y3 = self.conv3_2(x3)
        #capa4
        x4 = F.relu(self.conv1_1(y3))
        y4 = self.conv1_2(x4)
        #capa5
        x5 = F.relu(self.conv2_1(y4))
        y5 = self.conv2_2(x5)
        #capa6
        x6 = F.relu(self.conv3_1(y5))
        y6 = self.conv3_2(x6)
        return y1, y2, y3, y3, y4, y5, y6
# inicializa el writer
writer = SummaryWriter()

# tu modelo
model = HeatPropagationNet()

# una muestra de datos de entrada
inputs = torch.randn(1, 1, 7, 7)

# agrega el gráfico de tu modelo al writer
writer.add_graph(model, inputs)

# cierra el writer
writer.close()