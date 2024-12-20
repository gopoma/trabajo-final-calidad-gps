import sys
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def ChanVeseSegmentation(filepath, multiple=False):
    Image = cv2.imread(filepath, 1)
    image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    img = np.array(image, dtype=np.float64)

    IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
    IniLSF[30:80, 30:80] = -1
    IniLSF = -IniLSF

    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)



    #! plt.figure(1), plt.imshow(Image), plt.xticks([]), plt.yticks(
    #!     []
    #! )  # to hide tick values on X and Y axis
    #! plt.contour(IniLSF, [0], color="b", linewidth=2)
    #! plt.draw(), plt.show(block=False)



    def mat_math(intput, str):
        output = intput
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if str == "atan":
                    output[i, j] = math.atan(intput[i, j])
                if str == "sqrt":
                    output[i, j] = math.sqrt(intput[i, j])
        return output



    def CV(LSF, img, mu, nu, epison, step, show=False):
        Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
        Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan"))
        Iy, Ix = np.gradient(LSF)
        s = mat_math(Ix * Ix + Iy * Iy, "sqrt")
        Nx = Ix / (s + 0.000001)
        Ny = Iy / (s + 0.000001)
        Mxx, Nxx = np.gradient(Nx)
        Nyy, Myy = np.gradient(Ny)
        cur = Nxx + Nyy
        Length = nu * Drc * cur

        Lap = cv2.Laplacian(LSF, -1)
        Penalty = mu * (Lap - cur)

        s1 = Hea * img
        s2 = (1 - Hea) * img
        s3 = 1 - Hea
        C1 = s1.sum() / Hea.sum()
        C2 = s2.sum() / s3.sum()
        CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

        LSF = LSF + step * (Length + Penalty + CVterm)
        if show:
            plt.imshow(s, cmap="gray"), plt.show()
        return LSF



    mu = 1
    nu = 0.003 * 255 * 255
    #! num = 20
    num = 20
    epison = 1
    step = 0.1
    LSF = IniLSF



    info = [LSF]
    for i in range(1, num):
        LSF = CV(LSF, img, mu, nu, epison, step, False)
        info.append(LSF)
        # if i % 1 == 0:
        #     plt.imshow(Image), plt.xticks([]), plt.yticks([])
        #     plt.contour(LSF, [0], colors="r", linewidth=2)
        #     plt.draw(), plt.show(block=False), plt.pause(0.01)

    if not multiple:
        info = [info[-1]]

    return info
#!    print(len(info))

#!    counter = 0
#!    for lsf in info:
#!        plt.imshow(Image), plt.xticks([]), plt.yticks([])
#!        plt.contour(lsf, [0], colors="r", linewidth=2)
#!        plt.draw(), plt.show(block=False), plt.pause(0.5)
#!        plt.savefig(f"./tmp/img-result-{counter}")
#!        counter = counter + 1
#!        plt.cla()

    # ?fig, axarr = plt.subplots(2, 5)
    # ?plt.xticks([]), plt.yticks([])
    # ?for i in range(0, 2):
    # ?    for j in range(0, 5):
    # ?        axarr[i,j].imshow(Image)
    # ?        axarr[i, j].set_xticks([])  # Eliminar ticks del eje X
    # ?        axarr[i, j].set_yticks([])  # Eliminar ticks del eje Y
    # ?        axarr[i,j].contour(info[2 * i + j], [0], colors="r", linewidth=2)
    # ?
    # ?        fig.savefig("xd.jpg")
    # ?
    # ?plt.pause(2000)
