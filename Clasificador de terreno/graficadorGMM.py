import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Directorio base donde se encuentran tus archivos FFT y GMM
directorio_base = os.path.dirname(__file__)  # Obtener el directorio del script actual

# Define las rutas relativas desde el directorio base
ruta_carpeta_fft = os.path.join(directorio_base, 'FFT')
ruta_carpeta_gmm = os.path.join(directorio_base, 'GMM')

# Asumiendo que tienes un archivo por cada FFT y por cada GMM
archivo_fft = os.path.join(ruta_carpeta_fft, 'ade_rip_pend_gran_circulo_fftX.txt')
archivo_gmm = os.path.join(ruta_carpeta_gmm, 'ade_rip_pend_gran_circulo_gmmX.txt')


# Cargamos los datos de la FFT
# Suponemos que el archivo FFT tiene dos columnas, sin nombres de columna en el archivo
fft_data = pd.read_csv(archivo_fft, sep=" ", header=None, names=["Frequency (Hz)", "Amplitude"], comment='#')

# Cargamos los datos de GMM
gmm_data = pd.read_csv(archivo_gmm)

# Graficamos la FFT
plt.figure(figsize=(12, 7))
plt.plot(fft_data["Frequency (Hz)"], fft_data["Amplitude"], label='FFT Data')

# Calculamos la combinación no normalizada de las gaussianas ajustadas
combined_gaussian = np.zeros(fft_data.shape[0])
for i, row in gmm_data.iterrows():
    # Para cada gaussiana calculamos su PDF y lo sumamos al resultado
    gaussian = norm.pdf(fft_data["Frequency (Hz)"], row['media_x'], np.sqrt(row['cov_xx']))
    combined_gaussian += gaussian

# Graficamos la combinación lineal de las gaussianas
plt.plot(fft_data["Frequency (Hz)"], combined_gaussian, label='Combined Gaussian')

plt.title('FFT y Combinación Lineal de Gaussianas Ajustadas')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.show()
