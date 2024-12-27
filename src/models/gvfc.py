import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace as del2


def GVF(f, mu, ITER):
    """
    %GVF Calcula el flujo vectorial de gradiente.
    %   [u,v] = GVF(f, mu, ITER) calcula el
    %   GVF de un mapa de bordes f.  mu es el coeficiente de regularización del GVF
    %   e ITER es el número de iteraciones que se calcularán.
    """

    [m, n] = f.shape  # % Filas, columnas, canales

    fmin = np.min(f[:, :])
    fmax = np.max(f[:, :])
    f = (f - fmin) / (fmax - fmin)  # % Normaliza f al rango [0,1]

    f = BoundMirrorExpand(f)  # % Se encarga de la condición de frontera
    [fx, fy] = np.gradient(f)  # % Calcula el gradiente del mapa de bordes
    u = fx  # % Inicializa el GVF con el gradiente
    v = fy  # % Inicializa el GVF con el gradiente
    SqrMagf = fx * fx + fy * fy  # % Magnitud al cuadrado del campo de gradiente

    # % Resuelve iterativamente el GVF u,v
    for i in range(ITER):
        u = BoundMirrorEnsure(u)
        v = BoundMirrorEnsure(v)

        u = u + mu * 4 * del2(u) - SqrMagf * (u - fx)
        v = v + mu * 4 * del2(v) - SqrMagf * (v - fy)

    u = BoundMirrorShrink(u)
    v = BoundMirrorShrink(v)

    return [u, v]


def BoundMirrorEnsure(A):
    [m, n] = A.shape

    if m < 3 | n < 3:
        raise ("either the number of rows or columns is smaller than 3")

    yi = np.arange(0, m - 1)
    xi = np.arange(0, n - 1)
    B = A

    B[
        np.ix_(
            [
                1 - 1,
                m - 1,
            ],
            [
                1 - 1,
                n - 1,
            ],
        )
    ] = B[
        np.ix_(
            [
                3 - 1,
                m - 2 - 1,
            ],
            [
                3 - 1,
                n - 2 - 1,
            ],
        )
    ]
    # % mirror corners
    B[
        np.ix_(
            [
                1 - 1,
                m - 1,
            ],
            xi,
        )
    ] = B[
        np.ix_(
            [
                3 - 1,
                m - 2 - 1,
            ],
            xi,
        )
    ]
    # % mirror left and right boundary
    B[
        np.ix_(
            yi,
            [
                1 - 1,
                n - 1,
            ],
        )
    ] = B[
        np.ix_(
            yi,
            [
                3 - 1,
                n - 2 - 1,
            ],
        )
    ]
    # % mirror top and bottom boundary

    return B


def BoundMirrorExpand(A):
    """ """

    # shift for matlab style

    [m, n] = A.shape
    yi = np.arange(0, m + 1 - 1)
    # Genera conjunto de numeros entre valor inicio y final
    xi = np.arange(0, n + 1 - 1)

    B = np.zeros((m + 2, n + 2))
    # Retorna arreglo con zeros
    B[np.ix_(yi, xi)] = A
    # construye malla abierta con diferentes secuencias
    B[
        np.ix_(
            [
                1 - 1,
                m + 2 - 1,
            ],
            [
                1 - 1,
                n + 2 - 1,
            ],
        )
    ] = B[
        np.ix_(
            [
                3 - 1,
                m - 1,
            ],
            [
                3 - 1,
                n - 1,
            ],
        )
    ]
    # % mirror corners
    B[
        np.ix_(
            [
                1 - 1,
                m + 2 - 1,
            ],
            xi,
        )
    ] = B[
        np.ix_(
            [
                3 - 1,
                m - 1,
            ],
            xi,
        )
    ]
    # % mirror left and right boundary
    B[
        np.ix_(
            yi,
            [
                1 - 1,
                n + 2 - 1,
            ],
        )
    ] = B[
        np.ix_(
            yi,
            [
                3 - 1,
                n - 1,
            ],
        )
    ]
    # % mirror top and bottom boundary

    return B


def BoundMirrorShrink(A):
    [m, n] = A.shape
    yi = np.arange(0, m - 1)
    xi = np.arange(0, n - 1)
    B = A[np.ix_(yi, xi)]

    return B


if __name__ == "__main__":
    # Cargar la imagen
    filename = "car_3.bmp"
    image = cv2.imread(os.path.join("..", "assets", filename), cv2.IMREAD_GRAYSCALE)

    # Calcular GVF
    mu = 0.2  # Puedes ajustar este valor según tus necesidades
    ITER = 1  # Puedes ajustar este valor según tus necesidades
    u, v = GVF(image, mu, ITER)

    # Mostrar la imagen original y el resultado del GVF
    plt.figure(figsize=(10, 5))

    plt.subplot(131)
    plt.imshow(image, cmap="gray")
    plt.title("Imagen Original")

    plt.subplot(132)
    plt.quiver(u, v)
    plt.title("Campo GVF")

    plt.subplot(133)
    # plt.imshow(np.sqrt(u**2 + v**2), cmap='viridis')
    plt.imshow(np.sqrt(u**2 + v**2), cmap="gist_gray")
    plt.title("Magnitud del Campo GVF")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
