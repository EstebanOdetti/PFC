import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Asumiendo que tienes un archivo por cada FFT y por cada GMM
archivo_fft = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/ade_rip_pend_gran_circulo_fftX.txt'
archivo_gmm = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/ade_rip_pend_gran_circulo_gmmX.txt'

# Cargamos los datos de la FFT
# Suponemos que el archivo FFT tiene dos columnas, sin nombres de columna en el archivo
fft_data = pd.read_csv(archivo_fft, sep=" ", header=None, names=["Frequency (Hz)", "Amplitude"], comment='#')

# Cargamos los datos de GMM
gmm_data = pd.read_csv(archivo_gmm)

# Graficamos la FFT
plt.figure(figsize=(12, 7))
plt.plot(fft_data["Frequency (Hz)"], fft_data["Amplitude"], label='FFT Data')

# Graficamos las gaussianas ajustadas
for i, row in gmm_data.iterrows():
    # Calculamos el rango de la gaussiana
    gaus_x = np.linspace(fft_data["Frequency (Hz)"].min(), fft_data["Frequency (Hz)"].max(), 500)
    # Para cada gaussiana calculamos su PDF
    gaus_y = row['peso'] * norm.pdf(gaus_x, row['media_x'], np.sqrt(row['cov_xx']))
    plt.plot(gaus_x, gaus_y, label=f'Gaussian {i+1}')

plt.title('FFT y Gaussianas Ajustadas')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.show()
