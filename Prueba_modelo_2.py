import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el modelo previamente entrenado
modelo = tf.keras.models.load_model("modelo_formas.h5")

# Categorías del modelo (orden debe coincidir con el entrenamiento)
categorias = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle']

# Ruta a una imagen de prueba
ruta_imagen = "./Editable.png"
img = cv.imread(ruta_imagen)

# Preprocesamiento de la imagen
img = cv.resize(img, (80, 80))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = img / 255.0
img_ext = np.expand_dims(img, axis=0) # Agregar dimensión de lote

# Realizar la predicción
prediccion = modelo.predict(img_ext, verbose=0)
clase = np.argmax(prediccion) # Obtener índice de clase con mayor probabilidad
nombre_clase = categorias[clase]
confianza = np.max(prediccion) * 100 # Porcentaje de confianza

# Mostrar imagen con resultado de predicción
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(f"Predicción: {nombre_clase}\nConfianza: {confianza:.1f}%")
plt.axis('off')
plt.show()
