import cv2
import matplotlib
import matplotlib.pyplot as plt
from models.chan_vese_segmentation import ChanVeseSegmentation

filepath="hist_0.jpg"

info = ChanVeseSegmentation(filepath)
image = cv2.imread(filepath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

matplotlib.use('Agg')  # Use a non-interactive backend
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.contour(info[-1], [0], colors="r", linewidth=2)
plt.draw()
plt.show(block=False)
#!plt.pause(0.5)
plt.savefig(f"{filepath}-100.jpg")
