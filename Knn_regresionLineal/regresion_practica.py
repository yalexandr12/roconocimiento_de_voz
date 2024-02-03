# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 07:21:54 2023

@author: yalex
"""

# Importar las librerías necesarias
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd


# Definir las características de entrada y etiquetas de salida
# En este ejemplo, se utiliza una matriz de características de 20x10
# y un vector de etiquetas de salida con 5 clases posibles
X = np.random.rand(20, 10)
print("Matriz de características 20x10:", X)
y = np.random.randint(5, size=20)
print("Distribución de vector de etiquetas de clases:", y)
print("y:", y)

# Crear el dataframe con las dos matrices
df = pd.DataFrame(data=X, columns=['col'+str(i+1) for i in range(10)])
df['label'] = y

# Guardar el dataframe en un archivo csv
df.to_csv('datos.csv', index=False)

# Crear el modelo de regresión logística
lr = LogisticRegression()

# Entrenar el modelo utilizando los datos de entrada y salida
lr.fit(X, y)
# Utilizar el modelo entrenado para clasificar una nueva señal de voz
# En este ejemplo, se utiliza una señal de voz de tamaño 10x1
new_signal = np.random.rand(10, 1)
print("Señal generada:", new_signal)
predicted_label = lr.predict(new_signal.T)

# Imprimir la etiqueta de salida predicha
print("Etiqueta de salida predicha:", predicted_label[0])


