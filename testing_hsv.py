import os

import numpy as np
import cv2 as cv
import mahotas

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import label


def mean_hist():
    hist_h = []
    hist_s = []
    hist_v = []
    color = ('r', 'g', 'b')
    for filename in os.listdir("documentation/test"):
        img = cv.imread(os.path.join("documentation/test", filename))
        np.resize(img, (600, 300, 3))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        for i, col in enumerate(color):
            histogram = cv.calcHist([hsv], [i], None, [256], [0, 256])
            if i == 0:
                hist_h.append(histogram)
            elif i == 1:
                hist_s.append(histogram)
            else:
                hist_v.append(histogram)
    hist_h = np.asarray(hist_h).mean(axis=0)
    hist_s = np.asarray(hist_s).mean(axis=0)
    hist_v = np.asarray(hist_v).mean(axis=0)
    for i, col in enumerate(color):
        if i == 0:
            plt.plot(hist_h, color=col)
        elif i == 1:
            plt.plot(hist_s, color=col)
        else:
            plt.plot(hist_v, color=col)
    plt.legend(['H', 'S', 'V'])
    plt.title("Průměrný histogram")
    plt.ylabel("Frekvence")
    plt.xlabel("Hodnota složky")
    plt.axis()
    plt.show()


def make_hist(hsv):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histogram = cv.calcHist([hsv], [i], None, [256], [0, 256])
        plt.plot(histogram, color=col)
    plt.show()


def border(img, center):
    point = 0
    row = int(center[0])
    col = int(center[1])
    last_point = 0
    size = int(img.shape[1])

    for i in range(col, size):
        if img[row][i] == 0:
            point += 1
        else:
            last_point = i
            point = 0
        if point == int(size / 10):
            return last_point
    return int(size / 10)


def plot_orig_hsv(fignum, img, hsv):
    plt.figure(fignum)
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Původní obrázek")

    plt.subplot(2, 2, 2)
    plt.imshow(hsv)
    plt.title("HSV colorspace")

    # histogram puvodniho obrazku
    plt.subplot(2, 2, 3)
    make_hist(img)
    plt.title("Histogram RGB")

    plt.show()


def detect_face(binimg, origimg, center):
    plt.scatter(center[1], center[0], color='r')  # bod
    br = border(binimg, center)  # border
    size = int(abs(br - center[1]))
    rect = Rectangle((center[1] - size, center[0] - size), size * 2, size * 2, fill=False)
    plt.gca().add_patch(rect)
    plt.imshow(origimg)
    plt.show()


if __name__ == '__main__':

    plt.set_cmap('jet')  # set colormap
    path1 = "Images\man1.jpg"  # 3
    path2 = "Images\man3.jpg"  # 1
    mean_hist()
    for path in [path2, path1]:
        img = cv.imread(path)
        img_show = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # change color to HUE

        # plot_orig_hsv(1, img, hsv) # histogram a obrázek

        # Pouze HUE složka
        h_channel = hsv[:, :, 0]

        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.imshow(h_channel)
        plt.title("HUE složka")

        plt.subplot(2, 2, 2)
        h_channel[h_channel < 0] = 0
        h_channel[h_channel > 33] = 0
        h_channel[h_channel != 0] = 1
        plt.set_cmap('binary')
        plt.imshow(h_channel)
        plt.title("0-30 HUE")

        hsv2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        plt.subplot(2, 2, 3)
        min_HSV = np.array([0, 50, 30], dtype="uint8")
        max_HSV = np.array([30, 255, 255], dtype="uint8")  # min a max hodnoty hsv obličeje
        face = cv.inRange(hsv2, min_HSV, max_HSV)
        face[face > 0] = 1  # make binary
        center = mahotas.center_of_mass(face)  # center
        plt.imshow(face)
        plt.title("Aplikování prahových hodnot HSV")
        plt.xticks([])
        plt.yticks([])
        ######### MORFOLOGICKÉ OPERACE#########

        plt.figure(2)

        plt.set_cmap('binary')
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(face, kernel, iterations=1)
        plt.subplot(2, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(erosion)
        plt.title("Eroze")

        dilation = cv.dilate(face, kernel, iterations=1)
        plt.subplot(2, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(dilation)
        plt.title("Diletace")

        opening = cv.morphologyEx(face, cv.MORPH_OPEN, kernel)  # eroze následovaná diletací
        plt.subplot(2, 2, 3)
        plt.imshow(opening)
        plt.xticks([])
        plt.yticks([])
        plt.title("otevření")

        closing = cv.morphologyEx(face, cv.MORPH_CLOSE, kernel)  # diletace následovaná erozí
        plt.subplot(2, 2, 4)

        plt.imshow(closing)
        plt.xticks([])
        plt.yticks([])
        plt.title("uzavření")

        plt.show()
        ######### Labels #########
        plt.figure(3)
        plt.set_cmap('jet')
        lbl, ncc = label(opening)
        (values, counts) = np.unique(lbl, return_counts=True)
        max = np.argmax(np.bincount(lbl[lbl != 0]))
        lbl[lbl != max] = 0
        lbl[lbl != 0] = 1
        plt.subplot(1, 1, 1)
        plt.imshow(lbl)
        center = mahotas.center_of_mass(lbl)  # center
        plt.title("Detekovaný obličej")
        plt.figure(4)
        detect_face(lbl, img_show, center)
        br = border(lbl, center)  # border
        size = int(abs(br - center[1]))
        color = (255, 0, 0)
        stroke = 2
        cv.imshow("test", img)
        cv.waitKey()
        img = cv.rectangle(img, (int(center[1]) - size, int(center[0]) - size),
                           (int(center[1]) + size, int(center[0]) + size), color, stroke)
        cv.imshow("test", img)
        cv.waitKey()
        plt.show()
