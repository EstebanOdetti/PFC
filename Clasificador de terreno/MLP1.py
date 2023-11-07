from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# Load the dataset
file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/GMM/combined_terrain_data_one_hot_simplificado.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()

# Remove the columns not needed for the model
X = data.drop(columns=[col for col in data.columns if "one_hot_" in col])

# Get the one-hot encoded target labels
one_hot_columns = [col for col in data.columns if "one_hot_" in col]
y = data[one_hot_columns]

# Convert the one-hot encoded labels to a single column with categorical labels
y = y.idxmax(axis=1).str.replace('one_hot_', '')

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the transformations
(X_train.shape, X_test.shape), (y_train.shape, y_test.shape)

# Encode the target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert features and labels to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_train_tensor = torch.LongTensor(y_train_encoded)
y_test_tensor = torch.LongTensor(y_test_encoded)

# Create dataset and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 4  # You can adjust the batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.output_layer(x)
        return x

# Create an instance of the model
model = MLP(num_features=X_train.shape[1], num_classes=len(label_encoder.classes_))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 50 # You can adjust the number of epochs
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss}')

# Function to predict labels for the given loader
def predict(model, loader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_targets = []
    with torch.no_grad():  # No need to track the gradients
        for inputs, targets in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
    return np.array(all_preds), np.array(all_targets)

# Predict on test set
y_pred, y_true = predict(model, test_loader)

print(y_pred)
print("-------------")
print(y_true)
# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)
