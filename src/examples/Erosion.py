import cv2
import numpy as np

filename = "hist_0.jpg"

img = cv2.imread(f"../assets/{filename}", 0)

# Crear un kernel (matriz) de tamaño 2x2
# El kernel es una matriz utilizada para realizar operaciones de convolución en la imagen.
kernel = np.ones((2, 2), np.uint8)

# Aplicar la operación de erosión a la imagen
# 1. El primer parámetro es la imagen original.
# 2. El segundo parámetro es el kernel con el cual se realiza la convolución.
# 3. El tercer parámetro es el número de iteraciones, que determina cuántas veces aplicar la erosión.
img_erosion = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)

cv2.imwrite(f"../assets/eroded_{filename}", img_erosion)

cv2.waitKey(0)
