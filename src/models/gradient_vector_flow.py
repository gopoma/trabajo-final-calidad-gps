import sys

sys.path.append("../helpers")  # Agregar el directorio de utilidades predefinidas

import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.interpolate import splprep, splev, RectBivariateSpline

from contour import getContour


# Kernels de Sobel
# Aquí usamos coordenadas de imagen que coinciden con las coordenadas de arreglos
# en formato (x:altura, y:ancho)
SOBEL_X = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
SOBEL_Y = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])


# Kernel Laplaciano para la curvatura
LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _calcLaplacian(x):
    """Aplicar el operador Laplaciano."""
    return convolve(x, LAPLACIAN, mode="nearest")


def _calcGradient(x):
    """Devuelve el mapa de gradiente."""

    # Padding para evitar errores en los bordes
    pad = np.pad(x, pad_width=1, mode="edge")

    # Calcular las derivadas usando diferencias centradas
    gradx = 0.5 * (pad[2:, 1:-1] - pad[:-2, 1:-1])
    grady = 0.5 * (pad[1:-1, 2:] - pad[1:-1, :-2])

    return (gradx, grady)


def _getEdgeMap(u, sigma=2):
    """Devuelve un mapa de bordes usando el operador Sobel."""
    # Aplicar filtro Gaussiano para suavizar la imagen
    u = gaussian_filter(u, sigma=sigma)

    # Extraer bordes con el filtro Sobel
    fx = convolve(u, SOBEL_X, mode="nearest")
    fy = convolve(u, SOBEL_Y, mode="nearest")

    # Retornar la magnitud del gradiente
    return np.sqrt(0.5 * (fx**2.0 + fy**2.0))


class GVFSnake(object):
    """Snake basado en flujo vectorial de gradiente (GVF) para segmentación de imágenes.

    Parámetros:
    ----------------
    image: (H, W) ndarray
        Imagen de entrada.
    seed: (H, W) ndarray
        Semilla inicial.

    Resultados:
    ----------------
    Region: (H, W) ndarray
        Etiquetas de segmentación.
    """

    def __init__(
        self,
        alpha=0.01,   # Parámetro de continuidad
        beta=0.1,     # Parámetro de suavidad
        gamma=0.01,   # Paso artificial en el tiempo
        maxIter=1000, # Máximo número de iteraciones
        maxDispl=1.0, # Desplazamiento máximo permitido
        eps=0.1,      # Tolerancia para convergencia
        period=10,    # Periodo para el historial de energías
    ):
        # Parámetros del Modelo
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Parámetros Numéricos
        self.maxIter = int(maxIter)
        self.maxDispl = float(maxDispl)
        self.eps = eps

        # Historial de la evolución
        self.period = period
        self.xhistory = [0] * period
        self.yhistory = [0] * period

    def getGradientVectorFlow(
        self, fx, fy, CLF=0.25, mu=1.0, dx=1.0, dy=1.0, maxIter=1000, eps=1.0e-6
    ):
        """Calcular el flujo vectorial de gradiente (GVF)."""
        # Paso artificial bajo la restricción CFL
        dt = CLF * dx * dy / mu

        # Coeficientes para la ecuación
        b = fx**2.0 + fy**2.0
        c1, c2 = b * fx, b * fy

        currGVF = (fx, fy)  # Inicializar flujo vectorial con el gradiente del mapa de bordes
        for i in range(maxIter):
            # Evolucionar flujo
            nextGVF = (
                (1.0 - b * dt) * currGVF[0]
                + CLF * _calcLaplacian(currGVF[0])
                + c1 * dt,
                (1.0 - b * dt) * currGVF[1]
                + CLF * _calcLaplacian(currGVF[1])
                + c2 * dt,
            )

            # Actualizar flujo
            delta = np.sqrt(
                (currGVF[0] - nextGVF[0]) ** 2.0 + (currGVF[1] - nextGVF[1]) ** 2.0
            )
            if np.mean(delta) < eps:
                break
            else:
                currGVF = nextGVF

        return currGVF

    def run(self, image, seed):
        # Convertir la imagen de entrada al formato de punto flotante
        image = np.array(image, dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        # Obtener dimensiones de entrada
        if len(image.shape) == 2:
            height, width = image.shape
        elif len(image.shape) == 3:
            height, width, _ = image.shape
            image = np.mean(image, axis=-1)

        # Obtener el contorno inicial a partir de la región semilla
        contour = np.array(getContour(seed), dtype=np.float32)

        # Inicializar pivotes del snake
        tck, _ = splprep(contour.T, s=0)
        snake = splev(np.linspace(0, 1, 2 * len(contour)), tck)
        snake = np.array(snake).T.astype(np.float32)
        snake = np.array(contour).astype(np.float32)

        # Discretizar el snake
        xx, yy = snake[:, 0], snake[:, 1]
        for p in range(self.period):
            self.xhistory[p] = np.zeros(len(snake), dtype=np.float32)
            self.yhistory[p] = np.zeros(len(snake), dtype=np.float32)

        # Evaluar energías de la imagen
        edge = _getEdgeMap(image)

        # Evaluar el gradiente del mapa de bordes
        gradx, grady = _calcGradient(edge)

        # Obtener flujo vectorial de gradiente (GVF)
        GVF = self.getGradientVectorFlow(gradx, grady)

        # Obtener GVF continuo mediante interpolación
        xinterp = RectBivariateSpline(
            np.arange(height), np.arange(width), GVF[0], kx=2, ky=2, s=0
        )
        yinterp = RectBivariateSpline(
            np.arange(height), np.arange(width), GVF[1], kx=2, ky=2, s=0
        )

        # Construir matriz de forma del snake
        matrix = np.eye(len(snake), dtype=float)
        a = (
            np.roll(matrix, -1, axis=0) + np.roll(matrix, -1, axis=1) - 2.0 * matrix
        )  # Derivada de segundo orden
        b = (
            np.roll(matrix, -2, axis=0)
            + np.roll(matrix, -2, axis=1)
            - 4.0 * np.roll(matrix, -1, axis=0)
            - 4.0 * np.roll(matrix, -1, axis=1)
            + 6.0 * matrix
        )  # Derivada de cuarto orden
        A = -self.alpha * a + self.beta * b

        # Matriz inversa necesaria para el esquema numérico
        inv = np.linalg.inv(A + self.gamma * matrix).astype(np.float32)

        # Realizar optimización
        for step in range(self.maxIter):
            # Obtener valores de energía punto a punto
            fx = xinterp(xx, yy, grid=False).astype(np.float32)
            fy = yinterp(xx, yy, grid=False).astype(np.float32)

            # Evaluar nuevo snake
            xn = inv @ (self.gamma * xx + fx)
            yn = inv @ (self.gamma * yy + fy)

            # Confinar desplazamientos
            dx = self.maxDispl * np.tanh(xn - xx)
            dy = self.maxDispl * np.tanh(yn - yy)

            # Actualizar snake
            xx, yy = xx + dx, yy + dy

            # Verificar convergencia numérica
            index = step % (self.period + 1)
            if index < self.period:
                self.xhistory[index] = xx
                self.yhistory[index] = yy
            else:
                delta = np.max(
                    np.abs(np.array(self.xhistory) - xx)
                    + np.abs(np.array(self.yhistory) - yy),
                    axis=1,
                )
                if np.min(delta) < self.eps:
                    break

        # Marcar región del contorno snake en el mapa de píxeles
        seed = cv2.fillConvexPoly(seed, points=snake.astype(int), color=1)

        return seed, np.stack([xx, yy], axis=1)
