import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace as del2



def GVF(f, mu, ITER):
    """
    %GVF Compute gradient vector flow.
    %   [u,v] = GVF(f, mu, ITER) computes the
    %   GVF of an edge map f.  mu is the GVF regularization coefficient
    %   and ITER is the number of iterations that will be computed.
    """

    [m, n] = f.shape  # % Filas, cols, canales

    fmin = np.min(f[:, :])
    fmax = np.max(f[:, :])
    f = (f - fmin) / (fmax - fmin)
    # % Normalize f to the range [0,1]

    #! f = BoundMirrorExpand(f);  #% Take care of boundary condition
    f = BoundMirrorExpand(f)  # % Take care of boundary condition

    [fx, fy] = np.gradient(f)  # % Calculate the gradient of the edge map
    u = fx
    v = fy
    # % Initialize GVF to the gradient
    SqrMagf = fx * fx + fy * fy  # % Squared magnitude of the gradient field

    # % Iteratively solve for the GVF u,v
    for i in range(ITER):
        #! u = BoundMirrorEnsure(u)
        #! v = BoundMirrorEnsure(v)
        u = BoundMirrorEnsure(u)
        v = BoundMirrorEnsure(v)

        u = u + mu * 4 * del2(u) - SqrMagf * (u - fx)
        v = v + mu * 4 * del2(v) - SqrMagf * (v - fy)
        print(1, "%3d", i)
        if i % 20 == 0:
            print(1, "\n")

    print(1, "\n")

    #! u = BoundMirrorShrink(u);
    #! v = BoundMirrorShrink(v);
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



#! Cargar la imagen
# image = cv2.imread("bear.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("car_2.bmp", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("car_3.bmp", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("car_4.bmp", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("fighter.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("gourd.bmp", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("hist_0.jpg", cv2.IMREAD_GRAYSCALE)



#! Calcular GVF
# ? mu = 0.2  # Puedes ajustar este valor según tus necesidades
# ? ITER = 1  # Puedes ajustar este valor según tus necesidades
mu = 0.20  # Puedes ajustar este valor según tus necesidades
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
