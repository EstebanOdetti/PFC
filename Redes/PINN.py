import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import math
import matplotlib.pyplot as plt

#Tal cual hacemos en clase, definimos un MPL. 
class Net(torch.nn.Module):
    def __init__(self, input_features, size_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(input_features, size_hidden)
        self.hidden2 = nn.Linear(size_hidden, size_hidden)
        self.hidden3 = nn.Linear(size_hidden, size_hidden)
        self.hidden4 = nn.Linear(size_hidden, size_hidden)
        self.out = nn.Linear(size_hidden, n_output)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = torch.tanh(self.hidden4(x))
        x = self.out(x)        
        return x
    
# Definir la función de pérdida
def loss_fn(model, x_bc, y_bc, x_ic, y_ic):
    # Pérdida de la ecuación diferencial
    # requerimos gradiente
    x_ic.requires_grad = True
    # hacemos la prediccion
    y_hat = model(x_ic)
    # calculamos la derivada 
    dy_dx = grad(y_hat, x_ic, grad_outputs=torch.ones_like(y_hat), create_graph=True)[0]
    # calculamos la funcion de perdida
    loss_pde = torch.mean((torch.cos(x_ic) - dy_dx)**2)
  
    # Pérdida de las condiciones de contorno
    loss_bc = torch.mean((y_bc - model(x_bc))**2)

    # Pérdida total
    loss = loss_pde + loss_bc
    return loss
# Definir los datos de entrenamiento
# condifiones de borde

# Dispositivo en que se ejecturá el modelo: 'cuda:0' para GPU y 'cpu' para CPU
device = torch.device('cuda:0')

x_bc = torch.tensor([0.0, 12]).reshape(-1, 1)
y_bc = torch.tensor([0.0, torch.sin(torch.tensor([12]))]).reshape(-1, 1)
# datos de entrenamiento (entiendo que seria la "malla")
x_ic = torch.linspace(0.0, 12, 200).reshape(-1, 1)
y_ic = torch.zeros_like(x_ic)
# Entrenar la red neuronal 
net = Net(1,300,1)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

for i in range(10000):
    optimizer.zero_grad()
    x_bc = x_bc.to(device)
    y_bc = y_bc.to(device)
    x_ic = x_ic.to(device)
    y_ic = y_ic.to(device)
    loss = loss_fn(net, x_bc, y_bc, x_ic, y_ic)
    loss.backward()
    optimizer.step()
    # Print progress
    if i % 1000 == 0:
        print(f"Iteration {i}: Loss = {loss.item()}")


# Evaluar la red neuronal entrenada
x_test = torch.linspace(0, 12, 200).reshape(-1, 1)
y_test = torch.sin(x_test)
x_test = x_test.to(device)
y_test = y_test.to(device)
pred_test = net(x_test)

# Plot predicted and true solutions
plt.plot(x_test.cpu().detach().numpy(), y_test.cpu().detach().numpy(), 'r', label='True Solution')
plt.plot(x_test.cpu().detach().numpy(), pred_test.cpu().detach().numpy(), 'b', label='PINN Solution')
plt.legend()
plt.show()

