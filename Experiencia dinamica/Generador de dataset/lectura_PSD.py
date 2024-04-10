import os
import pandas as pd


directory_path = "C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Experiencia dinamica/Generador de dataset/experiencias/PSD TODAS FINAL"


files = []
for root, dirs, filenames in os.walk(directory_path):
    for filename in filenames:
        if "PSD" in filename:            
            files.append(os.path.join(root, filename))
            

combined_col1 = []
combined_col2 = []
combined_col3 = []


for file in files:

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:

            cols = line.strip().split(',')

            if len(cols) >= 2:        
                combined_col1.append(cols[0])
                combined_col2.append(cols[1])
                combined_col3.append(os.path.basename(file))


combined_data = pd.DataFrame({
    "Col1": combined_col1,
    "Col2": combined_col2,
    "Filename": combined_col3
})


combined_data.to_csv("combined_data_bien_ordenado.csv", index=False)
