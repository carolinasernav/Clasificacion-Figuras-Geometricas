# Clasificador de Figuras Geométricas

Este proyecto consiste en un clasificador de imágenes de figuras geométricas utilizando una red neuronal convolucional desarrollada con TensorFlow y OpenCV.

---

## Dataset

El dataset utilizado para entrenar y validar el modelo proviene de [Kaggle](#) _(https://www.kaggle.com/datasets/reevald/geometric-shapes-mathematics)_. Está dividido en carpetas para entrenar (train), validar (val) y ptobar(test) con categorías de las siguientes figuras:  
- circle  
- kite  
- parallelogram  
- rectangle  
- rhombus  
- square  
- trapezoid  
- triangle  

---

## Funcionamiento General del Modelo

El modelo es una red neuronal convolucional que recibe imágenes RGB de tamaño 80x80 píxeles. Está compuesta por varias capas convolucionales, max pooling, dropout y capas densas, y se entrena para clasificar imágenes en una de las 8 categorías geométricas mencionadas.

El entrenamiento se realiza durante 20 épocas con un batch size de 32, utilizando `sparse_categorical_crossentropy` como función de pérdida y el optimizador Adam.

Al final del entrenamiento, se guardan el modelo (`modelo_formas.h5`) y el historial de entrenamiento (`historial_entrenamiento.pkl`).

---

## Pruebas

### Visualización de Imágenes del Dataset

Se seleccionan imágenes de cada categoría para mostrar ejemplos del dataset, normalizadas y redimensionadas a 80x80 píxeles.

![Image_Alt]()

---

### Prueba de Clasificación en Imagen Individual

Se carga una imagen externa, se procesa y se usa el modelo para predecir la clase de figura geométrica, mostrando la predicción y el nivel de confianza.

*(Aquí puedes insertar una imagen del resultado de esta prueba)*

---

### Prueba de Clasificación en Múltiples Imágenes de Test

Se seleccionan varias imágenes aleatorias del conjunto de test, se clasifican con el modelo y se muestran junto con la clase real y la predicción del modelo con la confianza correspondiente.

Además, se grafican las métricas de pérdida y precisión tanto en entrenamiento como en validación a lo largo de las épocas.

*(Aquí puedes insertar una imagen de la prueba múltiple y las gráficas de entrenamiento)*

---

## Cómo Usar

1. Descargar el dataset desde Kaggle (link arriba).  
2. Organizar el dataset en carpetas `train`, `val` y `test` con subcarpetas por categoría.  
3. Ejecutar el script de entrenamiento para generar el modelo.  
4. Usar los scripts de prueba para validar el modelo con imágenes individuales o múltiples.

---

## Contacto

Si tienes preguntas o sugerencias, puedes contactarme a través de mi perfil de GitHub.

---

*Este repositorio es solo para mostrar el trabajo realizado en clasificación de figuras geométricas.*
