import cv2 as cv
import numpy as np
import os
import pickle
import tensorflow as tf

ruta_train = "./data/train" # Carpeta con imágenes de entrenamiento
ruta_val = "./data/val" # Carpeta con imágenes de validacion

# Categorias del dataset con los nombres de las subcarpetas
categorias = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle']
etiqueta = {cat: i for i, cat in enumerate(categorias)}

# Función para cargar y preprocesar imágenes
def cargar_datos(ruta_base, categorias, etiqueta):
    datos_x, datos_y = [], []
    for categoria in categorias:
        carpeta = os.path.join(ruta_base, categoria)
        for nombre_archivo in os.listdir(carpeta):
            ruta_img = os.path.join(carpeta, nombre_archivo)
            img = cv.imread(ruta_img)
            if img is None:
                continue
            img = cv.resize(img, (80, 80))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img / 255.0 
            datos_x.append(img)
            datos_y.append(etiqueta[categoria])
    return np.array(datos_x), np.array(datos_y)

# Carga de datos
x_train, y_train = cargar_datos(ruta_train, categorias, etiqueta)
x_val, y_val = cargar_datos(ruta_val, categorias, etiqueta)

# Definición del modelo CNN
Conv_modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(80, 80, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(len(categorias), activation='softmax')
])

# Compilación y entrenamiento
Conv_modelo.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

historial = Conv_modelo.fit(x_train, y_train,
                            epochs=20,
                            batch_size=32,
                            validation_data=(x_val, y_val))

#  Guardado del modelo y su historial
Conv_modelo.save("modelo_formas.h5") #? Guardar el Modelo

with open("historial_entrenamiento.pkl", "wb") as f: #? Guardar el historial del modelo
    pickle.dump(historial.history, f)