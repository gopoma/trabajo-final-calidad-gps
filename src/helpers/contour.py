from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt


def _isBoundary(pos, shape):
    """Verifica si una posición está en el borde de la imagen.

    Parámetros
    ----------------
    pos: (3,) list, tuple o ndarray
        Posición a verificar si está en el borde.

    shape: tuple
        Dimensiones de la imagen.

    Retorno
    ----------------
    boolean: bool
        Verdadero si la posición está en el borde, Falso en caso contrario.
    """
    # Verifica si la posición excede los límites de la imagen
    if pos[0] == -1 or pos[0] == shape[0]:
        return True
    if pos[1] == -1 or pos[1] == shape[1]:
        return True

    return False


def getContour(region, bClosed=True):
    """Obtiene el contorno de una región en la imagen.

    Parámetros
    ----------------
    region: (H, W) ndarray
        Región binaria (en escala de grises o binaria).

    bClosed: boolean, opcional
        Indica si el contorno debe estar cerrado (el primer y último punto serán iguales).
        Por defecto es True.

    Retorno
    ----------------
    contour: (N, 2) ndarray
        Puntos del contorno ordenados en dirección horaria.

    Notas
    ----------------
    - Los puntos del contorno están ordenados en sentido horario.
    """
    # Desplazamientos en sentido horario (vecindario de Moore)
    displi = [-1, -1, -1, 0, 1, 1, 1, 0]
    displj = [-1, 0, 1, 1, 1, 0, -1, -1]

    # Generar un mapa de bordes basado en la transformación de distancia
    edge = np.uint8(distance_transform_edt(region) == 1)

    # Obtener las dimensiones de la imagen
    height, width = edge.shape

    # Calcular el centro de la región
    center = (height // 2, width // 2)

    # Inicializar la consulta con la primera posición del borde encontrada
    xpos, ypos = np.argwhere(edge == 1)[0] # Encuentra el primer píxel de borde
    query = deque([(2, xpos, ypos)])       # (índice de vecino inicial, posición x, posición y)

    # Declarar un dominio de marcado para evitar duplicados
    mark = np.zeros_like(edge)

    # Lista para almacenar los puntos del contorno
    contour = []

    # Búsqueda de componentes conectados
    while query:
        # Obtener la posición actual
        start, i, j = query.popleft()

        # Girar los desplazamientos para empezar desde el vecino correcto
        dis = displi[start:] + displi[:start]
        djs = displj[start:] + displj[:start]

        # Encontrar el siguiente componente conectado en sentido horario
        for end, (di, dj) in enumerate(zip(dis, djs)):
            iq, jq = i + di, j + dj

            # Verificar si la posición está dentro de los límites y no ha sido marcada
            if _isBoundary((iq, jq), (height, width)):
                continue
            if not edge[iq, jq]:
                continue
            if mark[iq, jq]:
                continue
            else:
                mark[iq, jq] = 1 # Marcar la posición como visitada

            # Actualizar la consulta
            query.append(((start + end + 5) % 8, iq, jq))

            # Agregar el componente conectado a la lista de contorno
            contour.append((iq, jq))
            break

        # Si se ha recorrido todo el contorno, cerrarlo si es necesario
        if not query and bClosed:
            contour.append(contour[0])

    return contour
