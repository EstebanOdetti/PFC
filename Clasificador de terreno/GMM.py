import os
import numpy as np
from sklearn.mixture import GaussianMixture

# Define la ruta de la carpeta donde se encuentran tus archivos FFT
ruta_carpeta_fft = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/FFT'
ruta_carpeta_gmm = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/GMM'
# Lista todos los archivos en la carpeta
archivos_fft = [f for f in os.listdir(ruta_carpeta_fft) if f.endswith('_fftX.txt')]

# Itera sobre cada archivo
for archivo in archivos_fft:
    # Construye la ruta completa del archivo
    ruta_completa = os.path.join(ruta_carpeta_fft, archivo)

    # Lee los datos del archivo
    datos = np.loadtxt(ruta_completa, comments="#", delimiter=" ")
    frecuencias = datos[:, 0]  # Las frecuencias son la primera columna
    amplitudes = datos[:, 1]  # Las amplitudes son la segunda columna

    # Reshape de los datos para ajustar el modelo (si es necesario)
    X = np.column_stack((frecuencias, amplitudes))

    # Crea y ajusta el modelo GaussianMixture
    gmm = GaussianMixture(n_components=3, random_state=0, max_iter = 1000)
    gmm.fit(X)

    # Guardar los resultados
    nombre_archivo_gmm = archivo.replace('_fftX', '_gmmX')
    ruta_archivo_gmm = os.path.join(ruta_carpeta_gmm, nombre_archivo_gmm)

    with open(ruta_archivo_gmm, 'w') as archivo_gmm:
        archivo_gmm.write('media_x,media_y,cov_xx,cov_xy,cov_yy,peso\n')
        for i in range(gmm.n_components):
            mean_x, mean_y = gmm.means_[i]
            cov_xx, cov_xy, _, cov_yy = gmm.covariances_[i].flatten()
            peso = gmm.weights_[i]
            archivo_gmm.write(f"{mean_x},{mean_y},{cov_xx},{cov_xy},{cov_yy},{peso}\n")

print("Los modelos GMM han sido ajustados y guardados con éxito.")
