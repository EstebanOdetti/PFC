import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

def procesar_datos(file_path):
    # Cargar datos
    data = pd.read_csv(file_path, delimiter=',')

    # Extraer las columnas de entrada
    X = data[['Freq', 'PSD']].values

    # Organizar los datos en una matriz de 4x30
    X_reshape = np.empty((4, 30))
    X_reshape[0, :] = X[:30, 0]
    X_reshape[1, :] = X[:30, 1]
    X_reshape[2, :] = X[30:60, 1]
    X_reshape[3, :] = X[60:90, 1]

    return X_reshape

# Usar la función
file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/dataset_1_random_sin_nombre_exp_coma.csv'
matriz = procesar_datos(file_path)
print(matriz)

