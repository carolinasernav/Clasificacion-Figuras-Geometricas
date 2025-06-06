# Clasificador de Figuras Geométricas

Este proyecto consiste en un clasificador de imágenes de figuras geométricas utilizando una red neuronal convolucional desarrollada con TensorFlow y OpenCV.

---

## Dataset

El dataset utilizado para entrenar y validar el modelo proviene de [Kaggle](#) _(https://www.kaggle.com/datasets/reevald/geometric-shapes-mathematics)_. Está dividido en carpetas para entrenar (train), validar (val) y probar(test) con categorías de las siguientes figuras:  
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

<p align="center">
  <img src="https://github.com/user-attachments/assets/9deb0bec-d146-4c42-9620-fcc6fa6cc1b6" alt="Image">
</p>

---

## Pruebas

### Visualización de Imágenes del Dataset

Se seleccionan imágenes de cada categoría para mostrar ejemplos del dataset, normalizadas y redimensionadas a 80x80 píxeles.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f5ad6ae0-cce1-47d0-b041-252b380324df" alt="Image">
</p>

---

### Prueba de Clasificación en Imagen Individual

Se carga una imagen externa, se procesa y se usa el modelo para predecir la clase de figura geométrica, mostrando la predicción y el nivel de confianza.

<p align="center">
  <img src="https://github.com/user-attachments/assets/45adc758-e3a5-467d-ab07-661a3b2f1b15" alt="Image">
</p>

---

### Prueba de Clasificación en Múltiples Imágenes de Test

Se seleccionan varias imágenes aleatorias de la carpeta del dataset test, se clasifican con el modelo y posteriormente se muestran los resultados obtenidos con su confianza correspondiente, como se muestra a continuacion.

<p align="center">
  <img src="https://github.com/user-attachments/assets/99d5c756-773e-432b-83a3-32d22e2b0dae" alt="Image">
</p>

---

### Gráfias loss y Accuracy

Las gráficas muestran una reducción consistente en la pérdida tanto de entrenamiento como de validación, junto con un aumento sostenido en la precisión. Esto indica que el modelo aprendió correctamente a identificar las características de las figuras geométricas y generaliza bien a datos nuevos, sin señales evidentes de sobreajuste. Por tanto, se considera que el desempeño del modelo es sólido y confiable para esta tarea.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d870e961-5c6c-400a-a91e-b73e85a9b68b" alt="Image">
</p>
---
