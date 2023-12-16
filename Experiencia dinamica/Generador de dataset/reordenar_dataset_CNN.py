import pandas as pd
import numpy as np

def procesar_datos(file_path_x, file_path_y, output_file_path):
    # Cargar datos
    data_x = pd.read_csv(file_path_x, delimiter=',')    
    data_y = pd.read_csv(file_path_y, delimiter=',')
    num_blocks_x = data_x.shape[0] // 189
   # num_blocks_y = data_y.shape[0]
    print(num_blocks_x)
    # Inicializar una lista para almacenar las matrices transpuestas
    transposed_matrices = []

    # Procesar cada bloque de datos
    for i in range(num_blocks_x):
        # Extraer las filas correspondientes al bloque
        block_data_x = data_x.iloc[i*189 : (i+1)*189]
        #block_data_y = data_y.iloc[i : (i+1)]
        # Extraer las columnas de entrada
        X = block_data_x[['Freq', 'PSD']].values
        #Y = block_data_y[['P1_Frecuencia','P1_RMS']].values
        
        # Organizar los datos en una matriz de 4x30
        X_reshape = np.empty((4, 63))
        X_reshape[0, :] = X[:63, 0]
        X_reshape[1, :] = X[:63, 1]
        X_reshape[2, :] = X[63:126, 1]
        X_reshape[3, :] = X[126:189, 1]
       # X_reshape[4, :] = Y[0, 2]
       # X_reshape[5, :] = Y[1, 3]

        # Trasponer la matriz
        X_transposed = X_reshape.T

        # Añadir la matriz transpuesta a la lista
        transposed_matrices.append(X_transposed)

    # Concatenar las matrices transpuestas
    final_matrix = np.vstack(transposed_matrices)

    # Guardar la matriz final en un archivo CSV
    np.savetxt(output_file_path, final_matrix, delimiter=',')

# Usar la función
file_path_x = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/FINAL/combined_data.csv' 
file_path_y= 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/FINAL/Mediciones_simulaciones.csv'  
output_file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/matriz_total_ordenado_bien.csv'
procesar_datos(file_path_x, file_path_y, output_file_path)
