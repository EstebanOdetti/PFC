import os
import pandas as pd

# Directorio donde se encuentran tus archivos
folder_path = r"C:\Users\Usuario\Desktop\Proyectos\PyTorch\PyThorch Test\Clasificador de terreno\GMM"

# Función para procesar los archivos y unirlos
def process_and_combine_files(folder_path, terrain_name):
    # Crear rutas de archivo completo
    file_x = os.path.join(folder_path, f"{terrain_name}_gmmX.txt")
    file_y = os.path.join(folder_path, f"{terrain_name}_gmmY.txt")
    file_z = os.path.join(folder_path, f"{terrain_name}_gmmZ.txt")
    
    # Cargar los archivos
    df_x = pd.read_csv(file_x)
    df_y = pd.read_csv(file_y)
    df_z = pd.read_csv(file_z)
    
    # Renombrar las columnas para los ejes Y y Z para evitar duplicados al unir
    df_y.columns = [f"{col[:-2]}_y" if 'x' in col else col for col in df_y.columns]
    df_z.columns = [f"{col[:-2]}_z" if 'x' in col else col for col in df_z.columns]
    
    # Combinar los dataframes
    df_combined = pd.concat([df_x, df_y, df_z], axis=1)
    df_combined['nombre_terreno'] = terrain_name  # Añadir la columna con el nombre del terreno
    
    return df_combined

# Obtener una lista de nombres de terreno únicos basándose en los archivos del eje X
terrain_names = [f.replace('_gmmX.txt', '') for f in os.listdir(folder_path) if f.endswith('_gmmX.txt')]

# Procesar cada terreno y unir los archivos
all_combined_data = []  # Usar una lista para almacenar los DataFrames temporales

for terrain in terrain_names:
    combined_data = process_and_combine_files(folder_path, terrain)
    all_combined_data.append(combined_data)  # Añadir el DataFrame a la lista

# Concatenar todos los DataFrames en uno solo
combined_data = pd.concat(all_combined_data, ignore_index=True)

# Guardar el dataframe combinado en un nuevo archivo CSV
output_file = os.path.join(folder_path, 'combined_terrain_data.csv')
combined_data.to_csv(output_file, index=False)

print(f"El dataset combinado ha sido guardado en: {output_file}")
