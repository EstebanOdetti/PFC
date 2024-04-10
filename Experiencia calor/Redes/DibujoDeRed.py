from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch

class HeatPropagationNet(nn.Module):
    def __init__(self):
        super(HeatPropagationNet, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)


        self.conv2_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)


        self.conv3_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)


    def forward(self, x):

        x1 = F.relu(self.conv1_1(x))

        y1 = self.conv1_2(x1)

        x2 = F.relu(self.conv2_1(y1))
        y2 = self.conv2_2(x2)

        x3 = F.relu(self.conv3_1(y2))
        y3 = self.conv3_2(x3)

        x4 = F.relu(self.conv1_1(y3))
        y4 = self.conv1_2(x4)

        x5 = F.relu(self.conv2_1(y4))
        y5 = self.conv2_2(x5)

        x6 = F.relu(self.conv3_1(y5))
        y6 = self.conv3_2(x6)
        return y1, y2, y3, y3, y4, y5, y6

writer = SummaryWriter()


model = HeatPropagationNet()


inputs = torch.randn(1, 1, 7, 7)


writer.add_graph(model, inputs)


writer.close()