import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Cargar los datos
file_path = 'C:/Users/Usuario/Desktop/Proyectos/PyTorch/PyThorch Test/Clasificador de terreno/GMM/combined_terrain_data_one_hot_simplificado.csv'
data = pd.read_csv(file_path)

# Convertir las columnas "one_hot" en una única columna de clasificación multiclase
one_hot_columns = [col for col in data.columns if 'one_hot' in col]
data['terrain_type'] = data[one_hot_columns].idxmax(axis=1)

# Eliminar las columnas "one_hot" originales y la columna de clasificación del conjunto de datos
X = data.drop(one_hot_columns + ['terrain_type'], axis=1)
y = data['terrain_type']

# Dividir los datos en conjuntos de entrenamiento y prueba (usaremos una división 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el clasificador de Random Forest
rf_classifier = RandomForestClassifier(random_state=42, n_estimators = 10)

# Entrenar el modelo
rf_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = rf_classifier.predict(X_test)

print(y_pred)
print("--------------------------------")
print(y_test)
# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=1)

# Mostrar la precisión y el informe de clasificación
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
