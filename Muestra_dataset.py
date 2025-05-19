import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

ruta_train = "./data/train" # Carpeta con imágenes de entrenamiento  
train_x = []
train_y = []

# Nombres de las subcarpetas (categorías de figuras geométricas)
categorias = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle'] 

# Diccionario para mapear nombres de categoría a números
etiqueta = {}
for i, categoria in enumerate(categorias): #? Asignar etiquetas a cada tipo de figura
    etiqueta[categoria] = i

# Cargar y procesar las primeras 20 imágenes de cada categoría
for categoria in categorias:
    carpeta = os.path.join(ruta_train, categoria)  # Ruta de cada sub carpeta
    for i in os.listdir(carpeta)[:20]:  
        ruta_img = os.path.join(carpeta, i) # Ruta de cada imagen
        img = cv.imread(ruta_img)
        img = cv.resize(img, (80, 80))   # Redimensionar
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        img = img / 255.0 # Normalización de valores a [0,1]
        train_x.append(img)
        train_y.append(etiqueta[categoria]) # Asignar la etiqueta numérica

# Convertir las listas a arrays de NumPy
train_x = np.array(train_x)
train_y = np.array(train_y)

#  Mostrar 8 imágenes, una de cada categoría
fig, axes = plt.subplots(2, 4, figsize=(5, 5)) # Crear una cuadrícula de 2x4
axes = axes.flatten() # Aplanar los ejes para recorrer fácilmente
  
i = 0
category_count = {categoria: 0 for categoria in categorias}   # Llevar control de cuántas veces se muestra cada clase

for idx in range(8): 
    # Buscar una imagen de una categoría aún no mostrada
    while category_count[categorias[train_y[i]]] > 0:  
        i += 1
        if i >= len(train_y):  
            break
    if i >= len(train_y):
        break
     # Mostrar la imagen y su título
    category_count[categorias[train_y[i]]] += 1 
    axes[idx].imshow(train_x[i])
    axes[idx].set_title(categorias[train_y[i]])  
    i += 1

plt.tight_layout()
plt.show()