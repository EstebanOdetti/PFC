import os
import numpy as np

# Ruta a la carpeta con los archivos
carpeta = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/Experiencias todas'  # Asegúrate de poner la ruta correcta a tu carpeta aquí
carpeta_fft = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/FFT'
# Función para procesar y guardar FFT de cada archivo
def procesar_y_guardar_fft(archivo):
    tiempos = []
    datos_x = []
    datos_y = []
    datos_z = []

    with open(os.path.join(carpeta, archivo), 'r') as file:
        lines = file.readlines()
        for line in lines[2:]:  # Omitir la cabecera si la hay
            parts = line.strip().split(',')
            tiempos.append(float(parts[0]))
            datos_x.append(float(parts[1]))
            datos_y.append(float(parts[2]))
            datos_z.append(float(parts[3]))

    tiempos = np.array(tiempos)
    datos_x = np.array(datos_x)
    datos_y = np.array(datos_y)
    datos_z = np.array(datos_z)

    dt = np.mean(np.diff(tiempos))
    n = len(datos_x)
    frecuencias = np.fft.fftfreq(n, d=dt)

    # FFT para cada componente
    fft_x = np.fft.fft(datos_x)
    fft_y = np.fft.fft(datos_y)
    fft_z = np.fft.fft(datos_z)

    # Guardar FFTs en archivos
    nombre_base = archivo.split('.')[0]
    np.savetxt(os.path.join(carpeta_fft, f'{nombre_base}_fftX.txt'), np.column_stack((frecuencias[:n // 2], np.abs(fft_x)[:n // 2] * 2 / n)), header='Frequency (Hz), Amplitude')
    np.savetxt(os.path.join(carpeta_fft, f'{nombre_base}_fftY.txt'), np.column_stack((frecuencias[:n // 2], np.abs(fft_y)[:n // 2] * 2 / n)), header='Frequency (Hz), Amplitude')
    np.savetxt(os.path.join(carpeta_fft, f'{nombre_base}_fftZ.txt'), np.column_stack((frecuencias[:n // 2], np.abs(fft_z)[:n // 2] * 2 / n)), header='Frequency (Hz), Amplitude')

# Procesar cada archivo y guardar los resultados de la FFT
archivos = [archivo for archivo in os.listdir(carpeta) if archivo.endswith('.txt')]
for archivo in archivos:
    print(f'Procesando y guardando FFT de {archivo}...')
    procesar_y_guardar_fft(archivo)