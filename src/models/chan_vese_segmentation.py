import sys
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def ChanVeseSegmentation(filepath, multiple=False, num=20):
    # Cargar la imagen desde la ruta especificada
    Image = cv2.imread(filepath, 1)                 # Carga la imagen en color
    image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY) # Convierte la imagen a escala de grises
    img = np.array(image, dtype=np.float64)         # Convierte la imagen a un arreglo de tipo float64

    # Inicializar la función de nivel (Φ) para el método de conjuntos de nivel
    IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype) # Inicializa con valores positivos
    IniLSF[30:80, 30:80] = -1                                 # Define una región inicial con valores negativos
    IniLSF = -IniLSF                                          # Invierte la función de nivel

    # Convertir la imagen cargada a formato RGB para visualización
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)



    # Función auxiliar para operaciones matemáticas sobre matrices
    def mat_math(intput, str):
        output = intput
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if str == "atan":
                    output[i, j] = math.atan(intput[i, j])
                if str == "sqrt":
                    output[i, j] = math.sqrt(intput[i, j])
        return output



    # Método de evolución del contorno utilizando la ecuación de Chan-Vese
    def CV(LSF, img, mu, nu, epison, step, show=False):
        # Calcular la derivada regularizada y la función Heaviside suavizada
        Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
        Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan"))

        # Calcular gradiente de la función de nivel
        Iy, Ix = np.gradient(LSF)
        s = mat_math(Ix * Ix + Iy * Iy, "sqrt") # Magnitud del gradiente
        Nx = Ix / (s + 0.000001)                # Componente X del vector normalizado
        Ny = Iy / (s + 0.000001)                # Componente Y del vector normalizado
        Mxx, Nxx = np.gradient(Nx)
        Nyy, Myy = np.gradient(Ny)
        cur = Nxx + Nyy # Curvatura media

        # Calcular la longitud del contorno
        Length = nu * Drc * cur

        # Calcular término de penalización
        Lap = cv2.Laplacian(LSF, -1)
        Penalty = mu * (Lap - cur)

        # Calcular términos de energía de región
        s1 = Hea * img
        s2 = (1 - Hea) * img
        s3 = 1 - Hea
        C1 = s1.sum() / Hea.sum() # Promedio dentro del contorno
        C2 = s2.sum() / s3.sum()  # Promedio fuera del contorno
        CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

        # Actualizar la función de nivel
        LSF = LSF + step * (Length + Penalty + CVterm)

        if show:
            plt.imshow(s, cmap="gray"), plt.show()
        return LSF



    # Parámetros de evolución
    # mu = 1                 # Peso del término de suavidad
    # nu = 0.003 * 255 * 255 # Peso del término de longitud
    # num = 20               # Número de iteraciones
    # epison = 1             # Parámetro de regularización para Heaviside
    # step = 0.1             # Paso de evolución
    # LSF = IniLSF           # Inicializar la función de nivel

    # Parámetros de evolución
    mu = 1                 # Peso del término de suavidad
    nu = 0.003 * 255 * 255 # Peso del término de longitud
    # num = 20               # Número de iteraciones
    epison = 1             # Parámetro de regularización para Heaviside
    step = 0.1             # Paso de evolución
    LSF = IniLSF           # Inicializar la función de nivel


    # Evolución iterativa de la función de nivel
    info = [LSF]
    for i in range(1, num):
        LSF = CV(LSF, img, mu, nu, epison, step, False) # Actualizar LSF
        info.append(LSF) # Almacenar estado de la evolución

    # Devolver solo la última iteración si no se requiere almacenar todas
    if not multiple:
        info = [info[-1]]

    return info
