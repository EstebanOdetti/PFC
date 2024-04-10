import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
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
num_pruebas = total_casos - num_entrenamiento

temp_train_dirichlet = matriz_cargada_mezclada[:num_entrenamiento, :, :, 12]
temp_test_dirichlet = matriz_cargada_mezclada[num_entrenamiento:, :, :, 12]

temp_train = matriz_cargada_mezclada[:num_entrenamiento, :, :, 17]
temp_test = matriz_cargada_mezclada[num_entrenamiento:, :, :, 17]

temp_train_tensor = torch.from_numpy(temp_train).float()
temp_test_tensor = torch.from_numpy(temp_test).float()
temp_train_tensor_dirichlet = torch.from_numpy(temp_train_dirichlet).float()
temp_test_tensor_dirichlet = torch.from_numpy(temp_test_dirichlet).float()

temp_train_tensor = temp_train_tensor.unsqueeze(1)
temp_test_tensor = temp_test_tensor.unsqueeze(1)
temp_train_tensor_dirichlet = temp_train_tensor_dirichlet.unsqueeze(1)
temp_test_tensor_dirichlet = temp_test_tensor_dirichlet.unsqueeze(1)

train_dataset = TensorDataset(temp_train_tensor, temp_train_tensor)
test_dataset = TensorDataset(temp_test_tensor, temp_test_tensor)
train_dataset_dirichlet = TensorDataset(temp_train_tensor_dirichlet, temp_train_tensor)
test_dataset_dirichlet = TensorDataset(temp_test_tensor_dirichlet, temp_test_tensor)
class HeatPropagationNet(nn.Module):
    def __init__(self):
        super(HeatPropagationNet, self).__init__()
        

        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.5)


        self.conv2_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)


        self.conv3_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):

        x1 = self.leaky_relu(self.conv1_1(x))  
        y1 = self.leaky_relu(self.conv1_2(x1))  
        

        x2 = self.leaky_relu(self.conv1_1(y1))
        y2 = self.leaky_relu(self.conv1_2(x2))
        

        x3 = self.leaky_relu(self.conv2_1(y2))
        y3 = self.leaky_relu(self.conv2_2(x3))
        

        x4 = self.leaky_relu(self.conv2_1(y3))
        y4 = self.leaky_relu(self.conv2_2(x4))
        

        x5 = self.leaky_relu(self.conv3_1(y4))
        y5 = self.leaky_relu(self.conv3_2(x5))
        

        x6 = self.leaky_relu(self.conv3_1(y5))
        y6 = self.leaky_relu(self.conv3_2(x6))
        
        return y1, y2, y3, y4, y5, y6
    
def custom_loss(outputs, target):
    loss_tot = 0
    
    for output in outputs:
        loss_tot+=F.mse_loss(output, target)

    return loss_tot


writer = SummaryWriter('runs/experiment_1')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = HeatPropagationNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)


k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

best_val_loss = float('inf')
best_model_state = None
test_loss_values = []
val_loss_values = []
for fold, (train_index, val_index) in enumerate(kfold.split(train_dataset_dirichlet)):

    train_subset = torch.utils.data.Subset(train_dataset_dirichlet, train_index)
    val_subset = torch.utils.data.Subset(train_dataset_dirichlet, val_index)


    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=32)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=32)


    model = HeatPropagationNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)


    for epoch in range(100):
        train_loss_total, train_loss_ultima, val_loss_total, val_loss_ultima = 0.0, 0.0, 0.0, 0.0 
        num_batches_train, num_batches_val = 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_all = outputs[:-1]
            output_final = outputs[-1:]
            loss_final = custom_loss(output_final, labels)
            loss_all = custom_loss(outputs_all, labels)  + loss_final
            loss_all.backward()
            optimizer.step()
            train_loss_total += loss_all.item()
            train_loss_ultima += loss_final.item()
            num_batches_train += 1
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs_all = outputs[:-1]
                output_final = outputs[-1:]
                loss_all_validation = custom_loss(outputs_all, labels)
                loss_ultima_validation = custom_loss(output_final, labels)
                val_loss_total += loss_all_validation.item()
                val_loss_ultima += loss_ultima_validation.item()
                num_batches_val += 1

        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_model_state = model.state_dict()
        train_loss_total /= num_batches_train
        train_loss_ultima /= num_batches_train
        val_loss_total /= num_batches_val
        val_loss_ultima /= num_batches_val
        test_loss_values.append(train_loss_ultima)
        val_loss_values.append(val_loss_ultima)
        
        print(f'Fold {fold}, Epoch {epoch+1}, Training Loss (total): {train_loss_total}, Validation Loss: {val_loss_total}')
        print(f'Fold {fold}, Epoch {epoch+1}, Training Loss (ultima capa): {train_loss_ultima}, Validation Loss: {val_loss_ultima}')
      

plt.plot(test_loss_values)
plt.xlabel('Epoca')
plt.ylabel('Perdida de entrenamiento')
plt.title('Perdida de entrenamiento por Epoca')
plt.grid(True)
plt.show()

plt.plot(val_loss_values)
plt.xlabel('Epoca')
plt.ylabel('Perdida de Validación')
plt.title('Perdida de Validación por Epoca')
plt.grid(True)
plt.show()

model = HeatPropagationNet().to(device)
model.load_state_dict(best_model_state)

model = HeatPropagationNet().to(device)
model.load_state_dict(best_model_state)


test_dataset_dirichlet = TensorDataset(temp_test_tensor_dirichlet, temp_test_tensor)
test_loader = DataLoader(test_dataset_dirichlet, batch_size=32)


model.eval()

total_loss = 0
num_batches = 0


with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)


        outputs = model(inputs)


        loss = custom_loss(outputs, labels)
        total_loss += loss.item()
        num_batches += 1


mean_loss = total_loss / num_batches

print('Mean Loss on Test Set:', mean_loss)



fig, axs = plt.subplots(2, 2, figsize=(10, 20))


for j in range(2):

    im_output = axs[j, 0].imshow(outputs[-1][j, 0].cpu().numpy(), cmap='hot')
    axs[j, 0].set_title("Output de la red")
    fig.colorbar(im_output, ax=axs[j, 0], orientation='vertical')


    im_label = axs[j, 1].imshow(labels[j, 0].cpu().numpy(), cmap='hot')
    axs[j, 1].set_title("Ground Truth")
    fig.colorbar(im_label, ax=axs[j, 1], orientation='vertical')
    


fig.tight_layout(pad=5.0)


plt.show()


plt.figure()
ax = plt.gca()


ground_truth = labels[:, 0].cpu().numpy().flatten()
red_output = outputs[-1][:, 0].cpu().numpy().flatten()


plt.scatter(ground_truth, red_output, c='blue', label='Predicciones')


plt.scatter(ground_truth, ground_truth, c='red', label='Ground Truth', marker='x')


plt.xlabel("Ground Truth")
plt.ylabel("Predicciones")


plt.title("Dispersión de los puntos")


plt.legend()


plt.show()





