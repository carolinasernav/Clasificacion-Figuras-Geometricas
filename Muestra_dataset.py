import cv2 as cv,numpy as np,os,matplotlib.pyplot as plt

ruta_train = "E:/ESTUDIO/SEMESTRE 10-2/Sistemas Inteligentes/RN Convolucionales/dataset/train"  
train_x = []
train_y = []
categorias = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle'] 

etiqueta = {}
for i, categoria in enumerate(categorias): #? Asignar etiquetas a cada tipo de figura
    etiqueta[categoria] = i

for categoria in categorias:
    carpeta = os.path.join(ruta_train, categoria)  # Ruta de cada sub carpeta
    for i in os.listdir(carpeta)[:20]:  
        ruta_img = os.path.join(carpeta, i) # Ruta de cada imagen
        img = cv.imread(ruta_img)
        img = cv.resize(img, (80, 80))  
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        img = img / 255.0 # Normalizacion
        train_x.append(img)
        train_y.append(etiqueta[categoria]) 

train_x = np.array(train_x)
train_y = np.array(train_y)

fig, axes = plt.subplots(2, 4, figsize=(5, 5))
axes = axes.flatten()
  
i = 0
category_count = {categoria: 0 for categoria in categorias}  

for idx in range(8): 
    while category_count[categorias[train_y[i]]] > 0:  
        i += 1
        if i >= len(train_y):  
            break
    if i >= len(train_y):
        break
    category_count[categorias[train_y[i]]] += 1 
    axes[idx].imshow(train_x[i])
    axes[idx].set_title(categorias[train_y[i]])  
    i += 1

plt.tight_layout()
plt.show()