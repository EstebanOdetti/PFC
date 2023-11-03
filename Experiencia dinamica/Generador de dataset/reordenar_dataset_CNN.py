import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def procesar_datos(file_path, output_file_path):
    # Cargar datos
    data = pd.read_csv(file_path, delimiter=',')

    # Calcular el número de bloques de 90 filas
    num_blocks = data.shape[0] // 90

    # Inicializar una lista para almacenar las matrices transpuestas
    transposed_matrices = []

    # Procesar cada bloque de datos
    for i in range(num_blocks):
        # Extraer las filas correspondientes al bloque
        block_data = data.iloc[i*90 : (i+1)*90]

        # Extraer las columnas de entrada
        X = block_data[['Freq', 'PSD','frecuencia predominante','Tension resultante media']].values

        # Organizar los datos en una matriz de 4x30
        X_reshape = np.empty((6, 30))
        X_reshape[0, :] = X[:30, 0]
        X_reshape[1, :] = X[:30, 1]
        X_reshape[2, :] = X[30:60, 1]
        X_reshape[3, :] = X[60:90, 1]
        X_reshape[4, :] = X[:30, 2]
        X_reshape[5, :] = X[:30, 3]

        # Trasponer la matriz
        X_transposed = X_reshape.T

        # Añadir la matriz transpuesta a la lista
        transposed_matrices.append(X_transposed)

    # Concatenar las matrices transpuestas
    final_matrix = np.vstack(transposed_matrices)

    # Guardar la matriz final en un archivo CSV
    np.savetxt(output_file_path, final_matrix, delimiter=',')

# Usar la función
file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/dataset_1_random_sin_nombre_exp_coma.csv'
output_file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/matriz_total.csv'
procesar_datos(file_path, output_file_path)
