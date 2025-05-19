import cv2 as cv
import numpy as np
import os, random, pickle
import matplotlib.pyplot as plt
import tensorflow as tf

#  Cargar el modelo previamente entrenado
modelo = tf.keras.models.load_model("modelo_formas.h5")

# Categorías del modelo (orden debe coincidir con el entrenamiento)
categorias = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle']

# Ruta de las imagenes de la carpeta test del dataset
ruta_test = "./data/test"

# Recopilar rutas de imágenes y su clase real
imagenes_test = []
for categoria in categorias:
    carpeta = os.path.join(ruta_test, categoria)
    for archivo in os.listdir(carpeta):
        ruta_img = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta_img):
            imagenes_test.append((ruta_img, categoria))

# Seleccionar aleatoriamente entre 8 y 15 imágenes de prueba
cantidad = random.randint(8, 15)
imagenes_seleccionadas = random.sample(imagenes_test, cantidad)

# Realizar predicciones y visualizarlas
plt.figure(figsize=(15, 6))
for i, (ruta, real_categoria) in enumerate(imagenes_seleccionadas):
    img = cv.imread(ruta)
    img = cv.resize(img, (80, 80))
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    img_expandida = np.expand_dims(img_norm, axis=0)

    prediccion = modelo.predict(img_expandida, verbose=0)
    clase = np.argmax(prediccion)
    nombre_clase = categorias[clase]
    confianza = np.max(prediccion) * 100

    plt.subplot(3, 5, i + 1)
    plt.imshow(img_rgb)
    plt.title(f"Real: {real_categoria}\nPred: {nombre_clase}\n{confianza:.1f}%")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Cargar historial de entrenamiento (si existe) y graficar
with open("historial_entrenamiento.pkl", "rb") as f:
    historial = pickle.load(f)

# Graficar Pérdida (loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(historial['loss'], label='Pérdida entrenamiento')
plt.plot(historial['val_loss'], label='Pérdida validación')
plt.legend()
plt.title('Loss')

# Graficar Precisión (accuracy)
plt.subplot(1, 2, 2)
plt.plot(historial['accuracy'], label='Precisión entrenamiento')
plt.plot(historial['val_accuracy'], label='Precisión validación')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.show()