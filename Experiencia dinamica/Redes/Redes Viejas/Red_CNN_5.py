import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt


directorio_base = os.path.dirname(__file__)


file_path = os.path.join(
    directorio_base, "Datasets", "dataset_avanzado_random_en_lista.csv"
)
data = pd.read_csv(file_path, header=None)
data.columns = [
    "front_wheel_freq",
    "front_wheel_psdx",
    "front_wheel_psdy",
    "front_wheel_psdz",
    "front_target_freq",
    "front_target_ten",
]
wheel_data = data[
    ["front_wheel_freq", "front_wheel_psdx", "front_wheel_psdy", "front_wheel_psdz"]
].to_numpy()
targets = data[["front_target_freq"]].to_numpy()[::30]

wheel_data = wheel_data.reshape(-1, 30, 4)


wheel_data = torch.tensor(wheel_data, dtype=torch.float32).unsqueeze(1)
targets = torch.tensor(targets, dtype=torch.float32)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.combined_conv = nn.Conv2d(64, 1, (3, 3), padding=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(120, 1)

    def forward(self, wheel_data):
        x = self.conv1(wheel_data)
        x = self.conv2(x)
        x = self.combined_conv(x)
        x = self.flatten(x)
        output = self.fc(x)
        return output


model = CNNModel()


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


dataset = TensorDataset(wheel_data, targets)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)


n_epochs = 100
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (wheel_data_batch, targets_batch) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(wheel_data_batch)

        loss = criterion(outputs, targets_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")


model.eval()
with torch.no_grad():
    outputs = model(wheel_data)
    loss = criterion(outputs, targets)
    print("Loss: ", loss.item())


model.eval()
with torch.no_grad():
    outputs = model(wheel_data)


outputs = outputs.numpy()
targets = targets.numpy()


plt.figure(figsize=(8, 6))
plt.scatter(targets, targets, label="Objetivos reales (frecuencia)", c="blue")
plt.scatter(outputs, outputs, label="Predicciones (frecuencia)", c="red")
plt.xlabel("ground true")
plt.ylabel("prediccion")
plt.title("Gráfico de dispersión de frecuencia (Objetivos vs. Predicciones)")
plt.grid(True)
plt.legend()
plt.show()


input_example = torch.tensor(X_test[0], dtype=torch.float32).view(1, 1, -1)


torch.onnx.export(
    model,
    input_example,
    "model_CNN_dinamica_1.onnx",
    input_names=["input"],
    output_names=["output"],
)
