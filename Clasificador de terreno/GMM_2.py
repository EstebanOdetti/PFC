import os
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Directorio base donde se encuentran tus archivos FFT y GMM
directorio_base = os.path.dirname(__file__)  # Obtener el directorio del script actual

# Define las rutas relativas desde el directorio base
ruta_carpeta_fft = os.path.join(directorio_base, 'FFT')
ruta_carpeta_gmm = os.path.join(directorio_base, 'GMM')
# Lista todos los archivos en la carpeta
archivos_fft = [f for f in os.listdir(ruta_carpeta_fft) if f.endswith('_fftZ.txt')]

# Itera sobre cada archivo
for archivo in archivos_fft:
    # Construye la ruta completa del archivo
    ruta_completa = os.path.join(ruta_carpeta_fft, archivo)

    # Lee los datos del archivo
    datos = np.loadtxt(ruta_completa, comments="#", delimiter=" ")
    frecuencias = datos[1:, 0]  # Las frecuencias son la primera columna
    amplitudes = datos[1:, 1]  # Las amplitudes son la segunda columna

    # Reshape de los datos para ajustar el modelo (si es necesario)
    X = np.column_stack((frecuencias, amplitudes))

    # Crea y ajusta el modelo GaussianMixture
    gmm = GaussianMixture(
        n_components=3,
        tol=1e-9,
        max_iter=1000000
    )
    gmm.fit(X)

    # Guardar los resultados
    nombre_archivo_gmm = archivo.replace('_fftZ', '_gmmZ')
    ruta_archivo_gmm = os.path.join(ruta_carpeta_gmm, nombre_archivo_gmm)

    with open(ruta_archivo_gmm, 'w') as archivo_gmm:
        archivo_gmm.write('media_x_z,media_y_z,cov_xx_z,cov_xy_z,cov_yy_z,peso_z\n')
        for i in range(gmm.n_components):
            mean_x, mean_y = gmm.means_[i]
            cov_xx, cov_xy, _, cov_yy = gmm.covariances_[i].flatten()
            peso = gmm.weights_[i]
            archivo_gmm.write(f"{mean_x},{mean_y},{cov_xx},{cov_xy},{cov_yy},{peso}\n")
            
    # Visualización de la FFT
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(frecuencias, amplitudes)
    plt.title('FFT')

    # Visualización del modelo GMM
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), s=20, cmap='viridis')
    plt.title('Modelo GMM')
    plt.xlabel('Frecuencia')
    plt.ylabel('Amplitud')

   
    plt.close()

print("Los modelos GMM han sido ajustados y guardados con éxito.")
