
import pandas as pd

# Load the dataset
file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/GMM/combined_terrain_data.csv'
terrain_data = pd.read_csv(file_path)

# Perform one-hot encoding of the 'nombre_terreno' column
one_hot_encoded_data = pd.get_dummies(terrain_data['nombre_terreno'], prefix='one_hot')

# Add the one-hot encoded columns back to the original dataframe
terrain_data_one_hot = pd.concat([terrain_data, one_hot_encoded_data], axis=1)
# Define the path for the new CSV file
output_file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/GMM/combined_terrain_data_one_hot_simplificado.csv'

# Save the new dataframe with the one-hot encoded columns to a CSV file
terrain_data_one_hot.to_csv(output_file_path, index=False)
