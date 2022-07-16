import os
import time

import cv2
import cv2 as cv
import dlib
import mahotas
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label
from skimage.transform import resize

from testing_hsv import border
from skimage.feature import hog as hogs
from skimage.io import imread


def convert_and_trim_bb(image, rect):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    w = endX - startX
    h = endY - startY
    return startX, startY, w, h


def hog(img, detector, faces):
    if faces is None:
        faces = []
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    face = detector(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    boxes = [convert_and_trim_bb(img, r) for r in face]
    for (x, y, w, h) in boxes:
        color = (0, 255, 0)
        faces.append([x, y, x + w, y + h, color])
        height = y + h  # konečné souřadnice
        width = x + w
        stroke = 5
        img = cv.rectangle(img, (x, y), (width, height), color, stroke)
    return img


def viola_jones(img, detector, faces):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray_img, 1.3, 6)
    for (x, y, w, h) in boxes:
        color = (255, 0, 0)
        faces.append([x, y, x + w, y + h, color])
        height = y + h  # konečné souřadnice
        width = x + w
        stroke = 5
        img = cv.rectangle(img, (x, y), (width, height), color, stroke)
    return img


def hsv(img, faces):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    min_HSV = np.array([0, 50, 30], dtype="uint8")  # Nastavení trashholdu
    max_HSV = np.array([30, 255, 255], dtype="uint8")
    face = cv.inRange(hsv, min_HSV, max_HSV)  # vyříznutí pouze požadovaných hodnot
    face[face > 0] = 1  # binárně
    kernel = np.ones((4, 4), np.uint8)  # vytvoření jádra pro otevření
    opening = cv.morphologyEx(face, cv.MORPH_OPEN, kernel)  # otevření
    lbl, ncc = label(opening)  # labelování jednotlivých segmentů
    try:  # kontrola zda je možné labeling proběhl v pořádku
        max = np.argmax(np.bincount(lbl[lbl != 0]))  # největší lbl
    except ValueError:
        return img
    lbl[lbl != max] = 0  # nechat pouze největší lbl
    lbl[lbl != 0] = 1
    center = mahotas.center_of_mass(lbl)
    br = border(lbl, center)  # border
    size = int(abs(br - center[1]))  # vzdálenost od středu
    color = (0, 0, 255)
    x = int(center[1]) - size
    y = int(center[0]) - size
    w = int(center[1]) + size  # chyba upravit
    h = int(center[0]) + size
    faces.append([x, y, x + w, y + h, color])
    height = y + h  # konečné souřadnice
    width = x + w
    stroke = 5
    img = cv.rectangle(img, (int(center[1]) - size, int(center[0]) - size),
                       (int(center[1]) + size, int(center[0]) + size), color, stroke)
    return img


if __name__ == '__main__':
    PATH = "Osobnost/dbs_faces/"
    DET_HOG = dlib.get_frontal_face_detector()
    DET_VJ = cv.CascadeClassifier("data/viola_jones/haarcascade_frontalface_default.xml")
    test = True
    facess = []
    time1 = 0
    start = time.time()
    for person in os.listdir(PATH):  # Projdi každou osobu v zadané složce
        persons = os.listdir(PATH + person)
        for known_person in persons:
            unknown = cv2.imread(os.path.join(PATH, person, known_person))
            # img = imread(os.path.join(PATH, person, known_person))
            # resized_img = resize(img, (128 * 6, 64 * 8))

            # unknown2 = cv2.imread(os.path.join(PATH, person, known_person))
            # det1 = hog(unknown, DET_HOG, facess)
            # det1 = cv.cvtColor(det1, cv.COLOR_BGR2RGB)
            # plt.figure(1)
            # plt.subplot(1, 3, 1)
            # plt.imshow(det1)
            # plt.xticks([])
            # plt.yticks([])
            # plt.title("HOG")

            # plt.subplot(1, 3, 2)
            # det2 = viola_jones(unknown, DET_VJ, facess)
            # det2 = cv.cvtColor(det2, cv.COLOR_BGR2RGB)
            #
            # plt.imshow(det2)
            # plt.xticks([])
            # plt.yticks([])
            # plt.title("Viola-Jones")
            #
            # plt.subplot(1, 3, 3)
            # det3 = hsv(unknown2, facess) # unkown2
            # det3 = cv.cvtColor(det3, cv.COLOR_BGR2RGB)
            # plt.imshow(det3)
            # plt.xticks([])
            # plt.yticks([])
            # plt.title("Segmentace")
            # plt.show()
            # cv.waitKey()
            # cv.destroyAllWindows()
            # fd, hog_image = hogs(resized_img, orientations=9, pixels_per_cell=(32, 32),
            #                      cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            # plt.axis("off")
            # plt.imshow(hog_image, cmap="gray")
            # plt.show()
            # print("done")
    end = time.time()
    print("Trvání detekce HOG" + str(end - start))
