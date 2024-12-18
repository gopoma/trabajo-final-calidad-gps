import os
import sys
sys.path.append("../models")

import cv2
import matplotlib.pyplot as plt

from chan_vese_segmentation import ChanVeseSegmentation

def draw_multiple(filepath):
    info = ChanVeseSegmentation(filepath, True)

    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(0, len(info)):
        plt.title(f"Iteración {i + 1}")
        plt.imshow(image), plt.xticks([]), plt.yticks([])
        plt.contour(info[i], [0], colors="r", linewidth=2)
        plt.draw(), plt.show(block=False), plt.pause(0.5)

        if i is not len(info) - 1:
            plt.cla()

    plt.close()

    plt.title(f"Resultado Final")
    plt.imshow(image), plt.xticks([]), plt.yticks([])
    plt.contour(info[-1], [0], colors="r", linewidth=2)
    plt.draw(), plt.show(block=False), plt.pause(2000)

filename = "hist_0.jpg"
draw_multiple(os.path.join("..", "assets", filename))
