import csv
import os
import pickle
import time

import cv2
import cv2 as cv
import dlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import sort


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


def hog_detect_and_norm(img, detector):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    faces = detector(img)
    if not faces:  # není detekován žádný obličej
        return None
    x, y, w, h = [convert_and_trim_bb(img, r) for r in faces][0]  # Beru v potaz pouze jeden obličej
    face = img[y:y + h, x:x + w]
    face_resized = cv.resize(face, (64, 64), cv.INTER_AREA)
    return face_resized


def train_face_vector(path):
    DETECTOR = dlib.get_frontal_face_detector()

    names = []
    face_vec = []
    for person in os.listdir(path):  # Projdi každou osobu v zadané složce
        persons = os.listdir(path + person)
        for known_person in persons:
            img = cv.imread(os.path.join(PATH, person, known_person))
            face = hog_detect_and_norm(img, DETECTOR)
            if face is None:  # když není detekován obličej pokračuji další iterací
                continue
            names.append(person)
            face_g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)  # převedení do stupňů šedi
            face_img = face_g.reshape(4096, )  # převedení na sloupcový vektor
            face_vec.append(face_img)

    face_vec = np.asarray(face_vec)
    face_vectors = face_vec.T  # rozměr 90000x750
    w, wp = norm_face_vector(face_vectors)
    c = w.T @ w  # kovarianční matice - 750x750
    eigen_values, eigen_vectors = np.linalg.eig(c)  # vlastních čísel(750,) a vektorů (750x750), aut. sestupně
    e = w @ eigen_vectors  # 90000x15 vlastní prostor
    variance_explained = []

    for s in eigen_values:
        variance_explained.append((s / sum(eigen_values)) * 100)
    print(variance_explained)
    with open('testing/pca_variance', 'w', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f)
        for row in variance_explained:
            writer.writerow(row)
        # write a row to the csv file

    pi = e.T @ w  # projekce známých vektorů do vlastního prostoru 750x750
    data = {"pi": pi, "names": names, "mean_matrix": wp, "eigen_space": e}
    with open('data/datapca_traintest.pkl', 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    return data


def norm_face_vector(face_vectors):
    wp = face_vectors.mean(axis=1)  # průměrný vektor (90000,)
    wp = wp.reshape(face_vectors.shape[0], 1)  # (90000,1)
    return (face_vectors - wp), wp  # normalizovaný vektor


if __name__ == '__main__':
    PATH = "Osobnost/dbs_faces/"
    DET = dlib.get_frontal_face_detector()
    #k = 1
    if os.path.isfile('data/datapca_traintest.pkl'):
        with open('data/datapca_traintest.pkl', 'rb') as infile:
            data = pickle.load(infile)
    else:
        data = train_face_vector(PATH)

    ## Evaluating
    pi = data["pi"]  # 749x749
    wp = data["mean_matrix"]  # 90000x1
    names = data["names"]  # 749 rozeznaných
    e = data["eigen_space"]  # 90000x749
    for k in range(1, 12):
        start = time.time()
        print(k)
        recognized = 0
        unrecognized = 0
        for person in os.listdir(PATH):  # Projdi každou osobu v zadané složce
            persons = os.listdir(PATH + person)
            for known_person in persons:
                unknown = cv.imread(os.path.join(PATH, person, known_person))
                face = hog_detect_and_norm(unknown, DET)
                if face is None:  # když není detekován obličej pokračuji další iterací
                    unrecognized += 1
                    continue

                face_g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)  # převedení do stupňů šedi
                # plt.imshow(face_g)
                plt.set_cmap('gray')
                # plt.show()
                wpu = face_g.reshape(4096, 1)  # neznámý img
                #wu = wpu - wp  # odečtení vektorů od průměrného vektoru
                pt = e.T @ wpu  # projektce do našeho prostoru vlastních čísel

                #distance = np.sum(np.abs(pt - pi), axis=0)
                #distance = np.sum(pi * pt, axis=0) / np.sqrt(np.sum(np.power(pi,2), axis=0)*np.sum(np.power(pt,2), axis=0))
                distance = np.sum((np.power((pt - pi),2)), axis=0) # euklid. vzdálenost -k=3 - 54 k=2 -
                #idx = np.argpartition(distance, range(3))  # seřazené indexi dle vzdálenosti
                idx = np.argsort(distance)[1:k+1]
                name = "neznámé"
                counts = {}
                for i in idx:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                # print(idx)
                # print(np.argpartition(distance, 3))
                # print(idx[1])
                # print(np.argpartition(distance, 3)[1])
                # #print(names[np.argpartition(distance, 3)[1]])
                #print(names[idx[1]])
                #index = idx[1]
               # print(pi.shape)
                if name in person:
                    recognized += 1
                    # print("rec", recognized)
                else:
                    unrecognized += 1
                    # print("unrec", unrecognized)
        end = time.time()
        print("Trvání uložení souborů " + str(end - start))
        print("rozpoznáno: " + str(recognized))
        print("nerozpoznáno: " + str(unrecognized))
