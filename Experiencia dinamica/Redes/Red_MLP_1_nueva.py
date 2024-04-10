import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
import numpy as np


directorio_base = os.path.dirname(__file__)


combined_data_path = os.path.join(
    directorio_base, "Datasets\FINAL", "combined_data.csv"
)
simulations_data_path = os.path.join(
    directorio_base, "Datasets\FINAL", "Mediciones_simulaciones.csv"
)

combined_data_df = pd.read_csv(combined_data_path)
simulations_data_df = pd.read_csv(simulations_data_path)


X = combined_data_df[["Freq", "PSD"]].values
mediciones_por_experiencia = 63 * 3 * 2
y = np.repeat(
    simulations_data_df[["P1_Frecuencia", "P1_RMS"]].values,
    mediciones_por_experiencia,
    axis=0,
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = MLP().to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)


epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


model.eval()
with torch.no_grad():

    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f"Test Loss: {test_loss.item()}")
