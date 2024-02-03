# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:50:51 2023

@author: yalex
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('datos.csv')

# Convertir los datos de X a un arreglo de tipo float
X = np.array([list(map(float, row)) for row in df.iloc[:, :-1].values])

# Cargar los datos de y en un arreglo separado
y = np.array(df.iloc[:, -1].values)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el clasificador K-NN
knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")

# Entrenar el clasificador con los datos de entrenamiento
knn.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba
y_pred = knn.predict(X_test)

# Imprimir la matriz de confusión y el reporte de clasificación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))