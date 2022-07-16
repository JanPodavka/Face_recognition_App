import os
import pickle
import time

import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt

from pca import hog_detect_and_norm


def train_mace_filters(det, fpath, method):
    names = []
    h_maces = []
    X = None
    for person in os.listdir(fpath):  # Projdi každou osobu v zadané složce
        persons = os.listdir(fpath + person)
        for known_person in persons:
            img = cv2.imread(os.path.join(fpath, person, known_person))
            face = hog_detect_and_norm(img, det)
            if face is None:  # když není detekován obličej pokračuji další iterací
                continue
            face_g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            dft = np.fft.fft2(face_g)
            dft = np.fft.fftshift(dft)
            if X is None:
                X = dft.flatten()
                #   print(X)
            else:
                X = np.c_[X, dft.flatten()]

        #X = np.sort(X, axis=0)
        power_spec = np.mean((abs(X) ** 2), axis=1)
        d = np.diag(power_spec)
        d_inv = np.linalg.inv(d)
        if method in "mace:":
            xp = np.transpose(np.conjugate(X))
            u = np.ones((X.shape[1], 1))  # Nx1 elementů
            xdx = np.linalg.inv(xp @ d_inv @ X)
            h_mace = d_inv @ X @ xdx @ u  # Eavg
        elif method in "umace":
            h_mace = d_inv @ np.mean(X, axis=1)
        else:
            raise NameError("Špatně zvolená metoda")

        X = None
        h_mace = h_mace.reshape(64, 64)
        h_maces.append(h_mace)
        names.append(person)
    data = {"names": names, "mace_filtr": h_maces}
    with open('data\datamace_train.pkl', 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    return data


if __name__ == '__main__':

    img_unk = cv2.imread("Osobnost/dbs_faces/Adam_Sandler/Adam_Sandler (5).jpg")

    path = os.path.join("Osobnost/dbs_faces/")
    DET = dlib.get_frontal_face_detector()
    methods = "umace"
    if os.path.isfile('data\datamace_train.pkl'): #mace 3
        with open('data\datamace_train.pkl', 'rb') as infile:
            data = pickle.load(infile)
    else:
        data = train_mace_filters(DET, path, methods)

    print("++++++++++porovnání++++++++++++")
    start = time.time()
    maximum = 0
    name = "x"
    recognized = 0
    unrecognized = 0
    path = os.path.join("Osobnost/dbs_faces/")
    for person in os.listdir(path):  # Projdi každou osobu v zadané složce
        persons = os.listdir(path + person)
        for known_person in persons:
            unknown = cv2.imread(os.path.join(path, person, known_person))
            face = hog_detect_and_norm(unknown, DET)
            if face is None:  # když není detekován obličej pokračuji další iterací
                unrecognized += 1
                continue
            face_g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)  # převedení do stupňů šedi
            m = np.fft.fftshift(np.fft.fft2(face_g))
            MAX = 0
            for fil, person_label in zip(data["mace_filtr"], data["names"]):
                r = m * fil # korelace s filtrem
                inverse_fft = np.fft.ifft2(r)
                g = np.abs(inverse_fft)**2  # inverzní FFT
                r = abs(r) # Magnitudové spektrum
                # plt.imshow(r**2)
                # plt.show()
                #plt.imshow(abs(r))
                PSR = abs(np.max(r) - np.mean(r) / np.std(r))
                #plt.show()
                if PSR > MAX:
                    MAX = PSR
                    name = person_label

            MAX = 0
            if name in person:
                recognized += 1
                #print("rec", recognized)
            else:
                unrecognized += 1
                #print("unrec", unrecognized)
    end = time.time()
    print("Trvání uložení souborů " + str(end - start))
