import os
import pandas as pd

# Ruta del directorio donde se encuentran los archivos
directory_path = "C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Generador de dataset/experiencias/PSD TODAS FINAL"

# Encuentra todos los archivos en el directorio que contienen "PSD" en el nombre
files = []
for root, dirs, filenames in os.walk(directory_path):
    for filename in filenames:
        if "PSD" in filename:            
            files.append(os.path.join(root, filename))
            
# Inicializa listas vacías para las columnas combinadas
combined_col1 = []
combined_col2 = []
combined_col3 = []

# Itera sobre cada archivo y lee las dos primeras columnas
for file in files:
    # Lee el archivo
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Separa la línea en columnas usando coma como separador
            cols = line.strip().split(',')
            # Si hay más de una columna, añade la primera y segunda columna a las listas, y el nombre del archivo a la tercera lista
            if len(cols) >= 2:        
                combined_col1.append(cols[0])
                combined_col2.append(cols[1])
                combined_col3.append(os.path.basename(file))

# Crea un DataFrame con las columnas combinadas
combined_data = pd.DataFrame({
    "Col1": combined_col1,
    "Col2": combined_col2,
    "Filename": combined_col3
})

# Guarda el DataFrame combinado en un nuevo archivo
combined_data.to_csv("combined_data.csv", index=False)
