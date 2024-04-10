import pandas as pd
import numpy as np
import os


def procesar_datos(file_path_x, output_file_path):

    data_x = pd.read_csv(file_path_x, delimiter=",")
    num_blocks = data_x.shape[0] // 189

    processed_matrices = []

    for i in range(num_blocks):

        block = data_x.iloc[i * 189 : (i + 1) * 189]

        Freq = block["Freq"].iloc[:63].values
        Filename = block["Filename"].iloc[0]

        PSD_X = block["PSD"].iloc[:63].values
        PSD_Y = block["PSD"].iloc[63:126].values
        PSD_Z = block["PSD"].iloc[126:189].values

        combined_matrix = np.column_stack(
            [Freq, PSD_X, PSD_Y, PSD_Z, np.repeat(Filename, 63)]
        )
        processed_matrices.append(combined_matrix)

    final_matrix = np.vstack(processed_matrices)

    np.savetxt(output_file_path, final_matrix, delimiter=",", fmt="%s")


file_path_x = "C:/Users/Usuario/Desktop/Proyectos/PyTorch/PFC/Experiencia dinamica/Datasets/FINAL/combined_data.csv"

output_file_path = "C:/Users/Usuario/Desktop/Proyectos/PyTorch/PFC/Experiencia dinamica/Datasets/matriz_total_ordenado_bien.csv"
procesar_datos(file_path_x, output_file_path)
