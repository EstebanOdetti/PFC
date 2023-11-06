import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Directorio base donde se encuentran tus archivos FFT y GMM
directorio_base = os.path.dirname(__file__)

# Rutas de las carpetas FFT y GMM
ruta_carpeta_fft = os.path.join(directorio_base, 'FFT')
ruta_carpeta_gmm = os.path.join(directorio_base, 'GMM')

# Filtra los archivos FFT que específicamente terminan en '_fftX.txt'
archivos_fft_x = [f for f in os.listdir(ruta_carpeta_fft) if f.endswith('_fftX.txt')]

# Itera sobre cada archivo FFT que termina en '_fftX.txt'
for nombre_fft_x in archivos_fft_x:
    archivo_fft = os.path.join(ruta_carpeta_fft, nombre_fft_x)
    
    # Construye el nombre del archivo GMM correspondiente si sigue un patrón similar
    # Asumiendo que el nombre del archivo GMM es el mismo que el del archivo FFT pero termina en '_gmmX.txt'
    nombre_gmm_x = nombre_fft_x.replace('_fftX.txt', '_gmmX.txt')
    archivo_gmm = os.path.join(ruta_carpeta_gmm, nombre_gmm_x)
    print(archivo_gmm)
    print(archivo_fft)
    # Verifica si el archivo GMM existe antes de continuar
    if not os.path.exists(archivo_gmm):
        print(f"No se encontró el archivo GMM correspondiente: {nombre_gmm_x}")
        continue

    # Carga los datos de la FFT
    fft_data = pd.read_csv(archivo_fft, sep=" ", header=None, names=["Frequency (Hz)", "Amplitude"], comment='#')
    fft_data = fft_data.iloc[1:, :]

    # Carga los datos de GMM
    gmm_data = pd.read_csv(archivo_gmm)

    # Grafica la FFT
    plt.figure(figsize=(12, 7))
    plt.plot(fft_data["Frequency (Hz)"], fft_data["Amplitude"], label=f'FFT Data ({nombre_fft_x})')

    # Calcula la combinación no normalizada de las gaussianas ajustadas
    combined_gaussian = np.zeros(len(fft_data))
    for _, row in gmm_data.iterrows():
        gaussian = norm.pdf(fft_data["Frequency (Hz)"].astype(float), row['media_x'], np.sqrt(row['cov_xx']))
        combined_gaussian += gaussian

    # Grafica la combinación lineal de las gaussianas
    plt.plot(fft_data["Frequency (Hz)"], combined_gaussian, label=f'Combined Gaussian ({nombre_gmm_x})')

    plt.title(f'FFT y Combinación Lineal de Gaussianas Ajustadas')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.show()

print("Todas las gráficas han sido mostradas.")