import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio

import numpy as np
import matplotlib.pyplot as plt


placa = np.zeros((7, 7))

valores_sur = [20, 50]
tipo_borde = [[1, 1]]

for i in range(len(valores_sur)):
    if tipo_borde[0][0] == 1:

        valores_DIR = np.cos(np.linspace(0, 2*np.pi, 7)) * valores_sur[i]
        

        placa[-1, :] = valores_DIR
        

        plt.plot(range(7), valores_DIR, label=f'valores_sur = {valores_sur[i]}')


print(placa)

plt.xlabel('Nodos')
plt.ylabel('valores_DIR')
plt.title('Gráfico de la función coseno para el borde sur')
plt.legend()
plt.grid(True)
plt.show()


placa = np.zeros((7, 7))

valores_este = [20, 50]
tipo_borde = [[1, 1]]

for j in range(len(valores_este)):
    if tipo_borde[0][1] == 1:

        valores_DIR = (np.linspace(-1, 1, 7)**2) * valores_este[j]
        

        placa[:, -1] = valores_DIR
        

        plt.plot(range(7), valores_DIR, label=f'valores_este = {valores_este[j]}')


print(placa)

plt.xlabel('Nodos')
plt.ylabel('valores_DIR')
plt.title('Gráfico de la función cuadrática para el borde este')
plt.legend()
plt.grid(True)
plt.show()


placa = np.zeros((7, 7))

valores_norte = range(0, 101, 10)
tipo_borde = [[1, 1, 1]]

for o in range(len(valores_norte)):
    if tipo_borde[0][2] == 1:

        valores_DIR = np.ones(7) * valores_norte[o]
        

        placa[0, :] = valores_DIR
        

        plt.plot(range(7), valores_DIR, label=f'valores_norte = {valores_norte[o]}')


print(placa)

plt.xlabel('Nodos')
plt.ylabel('valores_DIR')
plt.title('Gráfico de los valores para el borde')
plt.legend()
plt.grid(True)
plt.show()


placa = np.zeros((7, 7))

valores_oeste = range(0, 101, 10)
tipo_borde = [[1, 1, 1, 1]]

for p in range(len(valores_oeste)):
    if tipo_borde[0][3] == 1:

        valores_DIR = np.ones(7) * valores_oeste[p]
        

        placa[:, 0] = valores_DIR
        

        plt.plot(range(7), valores_DIR, label=f'valores_oeste = {valores_oeste[p]}')


print(placa)

plt.xlabel('Nodos')
plt.ylabel('valores_DIR')
plt.title('Gráfico de los valores para el borde')
plt.legend()
plt.grid(True)
plt.show()

mat_fname = 'Datasets/mi_matriz_solo_diritletch_enriquesida.mat'
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
primeros_10_casos = temp_train_tensor[0:10]

for i in range(10):
    caso = primeros_10_casos[i]
    imagen = caso[:, :]
    plt.subplot(2, 5, i + 1)
    im = plt.imshow(caso, cmap='hot', origin='lower')
    plt.title(f'Caso {i+1}')


plt.tight_layout(rect=[0, 0, 0.9, 1])


cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Temperatura', rotation=270, labelpad=15)

plt.show()

