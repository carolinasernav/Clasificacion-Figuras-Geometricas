import cv2 as cv,numpy as np,tensorflow as tf, matplotlib.pyplot as plt

modelo = tf.keras.models.load_model("modelo_formas.h5")

categorias = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle']

ruta_imagen = "E:/ESTUDIO/SEMESTRE 10-2/Sistemas Inteligentes/RN Convolucionales/Editable.png"
img = cv.imread(ruta_imagen)

if img is None:
    raise FileNotFoundError(f"No se encontró la imagen en la ruta: {ruta_imagen}")

img = cv.resize(img, (80, 80))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = img / 255.0
img_ext = np.expand_dims(img, axis=0)

prediccion = modelo.predict(img_ext, verbose=0)
clase = np.argmax(prediccion)
nombre_clase = categorias[clase]
confianza = np.max(prediccion) * 100

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(f"Predicción: {nombre_clase}\nConfianza: {confianza:.1f}%")
plt.axis('off')
plt.show()
