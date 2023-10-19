import numpy as np

# Tamaño del dataset
dataset_size = 50  # Por ejemplo, 50 registros

# Entradas
# Generamos vectores aleatorios para las frecuencias en x, y, z para ambos acelerómetros
rear_wheel_acc = np.random.rand(dataset_size, 3)  # Valores entre 0 y 1, puedes multiplicar para cambiar el rango
front_wheel_acc = np.random.rand(dataset_size, 3)  # Valores entre 0 y 1

# Salidas
frequencies = np.random.randint(1, 257, dataset_size)  # Valores entre 1 y 256
tensions = np.random.uniform(20, 50, dataset_size)  # Valores entre 20 y 50

# Construimos el dataset
dataset = {
    'rear_wheel_acc': rear_wheel_acc,
    'front_wheel_acc': front_wheel_acc,
    'frequencies': frequencies,
    'tensions': tensions
}

# Imprimimos algunos datos para verificar
for i in range(5):
    print(f"Registro {i+1}:")
    print(f"Acelerómetro Rueda Trasera: {dataset['rear_wheel_acc'][i]}")
    print(f"Acelerómetro Rueda Delantera: {dataset['front_wheel_acc'][i]}")
    print(f"Frecuencia: {dataset['frequencies'][i]}")
    print(f"Tensión: {dataset['tensions'][i]:.2f} MPa")
    print("-" * 40)

