import os
import pickle

import cv2
import cv2 as cv
import face_recognition
import numpy as np

from face_rec import convert_and_trim_bb


def load_encodings():
    with open('data\data_train.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data  # pokud je dataset již vytvořený



def face_recogniton(training_data, img, faces):
    name = ""
    x, y, w, h = faces
    face = img[y:y + h, x:x + w]
    face = face[:, :, ::-1]
    small_face = cv.resize(face, (0, 0), fx=0.25, fy=0.25)
    #rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(small_face)
    if encoding:
        face_distance = face_recognition.face_distance(training_data["encodings"], encoding[0])
        small_dist_index = np.argmin(face_distance)
        name = training_data["names"][small_dist_index]
    return name

def load_mace():
    with open('data/datamace_train.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data  # pokud je dataset již vytvořený

def mace(training_data, img, faces):
    name = ""
    x, y, w, h = faces
    face = img[y:y + h, x:x + w]
    # face = face[:, :, ::-1]
    # small_face = cv.resize(face, (0, 0), fx=0.25, fy=0.25)
    #rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    face_g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)  # převedení do stupňů šedi
    face_g = np.resize(face_g, (64, 64))
    m = np.fft.fftshift(np.fft.fft2(face_g))
    MAX = 0
    for fil, person_label in zip(training_data["mace_filtr"], training_data["names"]):
        r = m * fil  # korelace s filtrem
        inverse_fft = np.fft.ifft2(r)
        g = np.abs(inverse_fft) ** 2  # inverzní FFT
        r = abs(r)  # Magnitudové spektrum
        PSR = abs(np.max(r) - np.mean(r) / np.std(r))
        # plt.show()
        if PSR > MAX:
            MAX = PSR
            name = person_label
    return name

def load_pca():
    with open('data/datapca_train.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data  # pokud je dataset již vytvořený

def pca(training_data, img, faces):
    name = ""
    k=3
    pi = training_data["pi"]  # 749x749
    wp = training_data["mean_matrix"]  # 90000x1
    names = training_data["names"]  # 749 rozeznaných
    e = training_data["eigen_space"]  # 90000x749
    x, y, w, h = faces
    face = img[y:y + h, x:x + w]
    # face = face[:, :, ::-1]
    # small_face = cv.resize(face, (0, 0), fx=0.25, fy=0.25)
    face_g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)  # převedení do stupňů šedi
    face_g = np.resize(face_g, (64, 64))
    wpu = face_g.reshape(4096, 1)  # neznámý img
    wu = wpu - wp  # odečtení vektorů od průměrného vektoru
    pt = e.T @ wu  # projektce do našeho prostoru vlastních čísel
    distance = np.sum(np.abs(pt - pi), axis=0)
    idx = np.argsort(distance)[0:k + 1]
    name = "neznámé"
    counts = {}
    for i in idx:
        name = names[i]
        counts[name] = counts.get(name, 0) + 1
        name = max(counts, key=counts.get)
    return name

if __name__ == '__main__':
    pass
