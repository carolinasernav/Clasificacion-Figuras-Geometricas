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

![Image Alt](https://github.com/carolinasernav/Clasificacion-Figuras-Geometricas/blob/7eae8123c5d62f948414341e8342f5910a37ed86/Epoch.png)

---

## Pruebas

### Visualización de Imágenes del Dataset

Se seleccionan imágenes de cada categoría para mostrar ejemplos del dataset, normalizadas y redimensionadas a 80x80 píxeles.

![Image_Alt](https://github.com/carolinasernav/Clasificacion-Figuras-Geometricas/blob/83416b406f7fc83d923bf911fa61906b9ba9505b/Imagenes_Dataset.jpg)

---

### Prueba de Clasificación en Imagen Individual

Se carga una imagen externa, se procesa y se usa el modelo para predecir la clase de figura geométrica, mostrando la predicción y el nivel de confianza.

<p align="center">
  <img src="https://github.com/carolinasernav/Clasificacion-Figuras-Geometricas/blob/7eae8123c5d62f948414341e8342f5910a37ed86/Prueba%20Individual.png)">
</p>
---

### Prueba de Clasificación en Múltiples Imágenes de Test

Se seleccionan varias imágenes aleatorias de la carpeta del dataaset test, se clasifican con el modelo y posteriormente se muestran los resultados obtenidos con su confianza correspondiente, como se muestra a continuacion.
![Image_Alt](https://github.com/carolinasernav/Clasificacion-Figuras-Geometricas/blob/982cad174d8e1b061926a66b91ecdfcf68a6d928/Resultados_P1.png)

---

### Gráfias loss y Accuracy

Las gráficas muestran una reducción consistente en la pérdida tanto de entrenamiento como de validación, junto con un aumento sostenido en la precisión. Esto indica que el modelo aprendió correctamente a identificar las características de las figuras geométricas y generaliza bien a datos nuevos, sin señales evidentes de sobreajuste. Por tanto, se considera que el desempeño del modelo es sólido y confiable para esta tarea.

![Image_Alt](https://github.com/carolinasernav/Clasificacion-Figuras-Geometricas/blob/982cad174d8e1b061926a66b91ecdfcf68a6d928/Graficas.png)

---
