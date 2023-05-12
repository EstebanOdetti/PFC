import deepxde as dde 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

datos_totales = pd.read_csv(r'C:\Users\Usuario\Desktop\PyThorch Test\datos_sinteticos_ver_3.csv')

# Seleccionamos las columnas deseadas
columnas_deseadas = ["x", "y", "borde", "model.k", "model.G", "model.c", "BN", "BS", "Be", "Bo", "columna_DIR1", "columna_DIR2", "columna_NEU1", "columna_NEU2", "columna_ROB1", "columna_ROB2"]
datos = datos_totales[columnas_deseadas]

# Seleccionamos la columna PHI_temp
datos_PHI_temp = datos_totales["PHI_temp"]

# Seleccionamos las columnas Q_tempx y Q_tempy
datos_Q_tempx = datos_totales["Q_tempx"]
datos_Q_tempy = datos_totales["Q_tempy"]

# Usamos train test split para hacer esquema 80-20
datos_tr, datos_ts, salida_esperada_tr, salida_esperada_ts = train_test_split(datos, datos_PHI_temp, test_size=0.2,
                                                                              random_state=0)


#primero a numpy
datos_tr = datos_tr.to_numpy()
datos_ts =datos_ts.to_numpy()
salida_esperada_tr =salida_esperada_tr.to_numpy()
salida_esperada_ts =salida_esperada_ts.to_numpy()


