import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

datos_totales = pd.read_csv(r'C:\Users\Usuario\Desktop\PyThorch Test\datos_sinteticos_ver_3.csv')

# Seleccionamos las columnas deseadas
columnas_deseadas = ["x", "y", "borde", "model.k", "model.G", "model.c", "BN", "BS", "Be", "Bo", "columna_DIR1", "columna_DIR2", "columna_NEU1", "columna_NEU2", "columna_ROB1", "columna_ROB2"]
datos = datos_totales[columnas_deseadas]

# Seleccionamos la columna PHI_temp
datos_PHI_temp = datos_totales["PHI_temp"]

# Seleccionamos las columnas Q_tempx y Q_tempy
datos_Q_tempx = datos_totales["Q_tempx"]
datos_Q_tempy = datos_totales["Q_tempy"]

# Usamos train test split para hacer esquema 80-20
datos_tr, datos_ts, salida_esperada_tr, salida_esperada_ts = train_test_split(datos, datos_PHI_temp, test_size=0.2,
                                                                              random_state=0)


#primero a numpy
datos_tr = datos_tr.to_numpy()
datos_ts =datos_ts.to_numpy()
salida_esperada_tr =salida_esperada_tr.to_numpy()
salida_esperada_ts =salida_esperada_ts.to_numpy()
#usamos StandardScaler para normalizar los datos!
scaler = StandardScaler()
scaler = scaler.fit(datos_tr)
datos_tr = scaler.transform(datos_tr)
scaler = scaler.fit(datos_ts)
datos_ts = scaler.transform(datos_ts)
scaler = scaler.fit(salida_esperada_tr.reshape(-1, 1))
salida_esperada_tr = scaler.transform(salida_esperada_tr.reshape(-1, 1))
scaler = scaler.fit(salida_esperada_ts.reshape(-1, 1))
salida_esperada_ts = scaler.transform(salida_esperada_ts.reshape(-1, 1))


#Tal cual hacemos en clase, definimos un MPL. 
class NetMLP(torch.nn.Module):
    def __init__(self, input_features, size_hidden, n_output):
        super(NetMLP, self).__init__()
        
        self.hidden1 = nn.Linear(input_features, size_hidden)
        self.hidden2 = nn.Linear(size_hidden, size_hidden)
        self.hidden3 = nn.Linear(size_hidden, size_hidden)
        self.hidden4 = nn.Linear(size_hidden, size_hidden)
        self.hidden5 = nn.Linear(size_hidden, size_hidden)
        self.out = nn.Linear(size_hidden, n_output)

    def forward(self, x):
        x = F.rrelu(self.hidden1(x))
        x = F.rrelu(self.hidden2(x))
        x = F.rrelu(self.hidden3(x))
        x = F.rrelu(self.hidden4(x))
        x = F.rrelu(self.hidden5(x))
        x = self.out(x)        
        return x
# Dispositivo en que se ejecturá el modelo: 'cuda:0' para GPU y 'cpu' para CPU
device = torch.device('cuda:0')

# Creamos un loader iterable indicandole que debe leer los datos a partir de
# del dataset creado en el paso anterior. Este objeto puede ser iterado
# y nos devuelve de a un batch (x, y).

salida_esperada_tr = torch.tensor(salida_esperada_tr).unsqueeze(1)
dataset = TensorDataset(torch.from_numpy(datos_tr).clone().float(), salida_esperada_tr.float())
loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
datos_tr = torch.tensor(datos_tr).float()
datos_ts = torch.tensor(datos_ts).float()

salida_esperada_tr = torch.tensor(salida_esperada_tr).float()
salida_esperada_ts = torch.tensor(salida_esperada_ts).float()

# Entrenar la red neuronal 
net = NetMLP(16,500,1)
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion =nn.MSELoss()
loss_list = []
for i in range(100000):
    total_loss = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        loss =criterion(net(x),y)
        # Realizo la pasada backward por la red        
        loss.backward()
        
        # Actualizo los pesos de la red con el optimizador
        optimizer.step()

        # Me guardo el valor actual de la función de pérdida para luego graficarlo
        loss_list.append(loss.data.item())

        # Acumulo la loss del minibatch
        total_loss += loss.item() * y.size(0)

    # Normalizo la loss total   
    total_loss/= len(loader.dataset)


    print('Epoch %d, loss = %g' % (i, total_loss))
        
plt.figure()
loss_np_array = np.array(loss_list)
plt.plot(loss_np_array, alpha = 0.3)
N = 60
running_avg_loss = np.convolve(loss_np_array, np.ones((N,))/N, mode='valid')
plt.plot(running_avg_loss, color='red')
plt.title("Función de pérdida durante el entrenamiento")
plt.show()

# Definimos un método para mostrar las predicciones como un scatter plot 
# y graficamos la recta de regresión para esos datos.
def plotScatter(x_data, y_data, title, fit_line=True):
    plt.figure()

    plt.plot(x_data, y_data, '+')
    plt.xlabel('Valor real')
    plt.ylabel('Predicción')
    plt.title(title)

    if fit_line:
        X, Y = x_data.reshape(-1,1), y_data.reshape(-1,1)
        plt.plot( X, LinearRegression().fit(X, Y).predict(X) )
    plt.show()    

# Dibujamos el ground truth vs las predicciones en los datos de entrenamiento

py = net(torch.FloatTensor(datos_tr.cpu()).to(device))
y_pred_train = py.cpu().detach().numpy()
plotScatter(salida_esperada_tr, y_pred_train, "Training data")

# Dibujamos el ground truth vs las predicciones en los datos de test
py = net(torch.FloatTensor(datos_ts.cpu()).to(device))
y_pred_test = py.cpu().detach().numpy()
plotScatter(salida_esperada_ts, y_pred_test, "Test data")

print ("MSE medio en training: " + str(((salida_esperada_tr - y_pred_train)**2).mean()))
print ("MSE medio en test: " + str(((salida_esperada_ts - y_pred_test)**2).mean()))