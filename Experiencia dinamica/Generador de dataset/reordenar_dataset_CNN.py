import pandas as pd
import numpy as np
import os


def procesar_datos(file_path_x, output_file_path):
    # Cargar datos
    data_x = pd.read_csv(file_path_x, delimiter=',')
    num_blocks = data_x.shape[0] // 189

    # Inicializar una lista para almacenar las matrices procesadas
    processed_matrices = []

    # Procesar cada bloque de datos
    for i in range(num_blocks):
        # Extraer las filas correspondientes al bloque
        block = data_x.iloc[i*189 : (i+1)*189]

        # Extraer las columnas de Freq y Filename
        Freq = block['Freq'].iloc[:63].values
        Filename = block['Filename'].iloc[0]

        # Extraer y organizar los datos de PSD para cada eje
        PSD_X = block['PSD'].iloc[:63].values
        PSD_Y = block['PSD'].iloc[63:126].values
        PSD_Z = block['PSD'].iloc[126:189].values

        # Combinar las columnas en una matriz y añadirla a la lista
        combined_matrix = np.column_stack([Freq, PSD_X, PSD_Y, PSD_Z, np.repeat(Filename, 63)])
        processed_matrices.append(combined_matrix)

    # Concatenar todas las matrices procesadas
    final_matrix = np.vstack(processed_matrices)

    # Guardar la matriz final en un archivo CSV
    np.savetxt(output_file_path, final_matrix, delimiter=',', fmt='%s')

# Usar la función
file_path_x = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/FINAL/combined_data.csv' 

output_file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Datasets/matriz_total_ordenado_bien.csv'
procesar_datos(file_path_x, output_file_path)
