import pickle
import time

import cv2
import face_recognition
import cv2 as cv
import os
import dlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


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


def get_encogings(path):
    images1 = []
    names = []
    start = time.time()
    if os.path.isfile('data\data_train.pkl'):
        with open('data\data_train.pkl', 'rb') as infile:
            data = pickle.load(infile)

        return data  # pokud je dataset již vytvořený
    for person in os.listdir(path):  # Projdi každou osobu v zadané složce
        persons = os.listdir(path + person)
        for known_person in persons:
            img = cv.imread(PATH + person + "/" + known_person)
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            enc = face_recognition.face_encodings(rgb, boxes)
            if enc:
                for encoding in enc:  # enc detekoval žádný obličej
                    images1.append(encoding)
                    names.append(person)
            else:
                print("Žádný nalezený obličej " + known_person)

    end = time.time()
    print("Trvání načtení souborů " + str(end - start))
    # Uložení dat do souborů pickle
    data = {"encodings": images1, "names": names}
    start = time.time()
    with open('data\data_train.pkl', 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print("Trvání uložení souborů " + str(end - start))
    return data



def recognize(training_data, path, method, detector, plot):
    start = time.time()
    recognized = 0
    unrecognized = 0
    k = 2
    for person in os.listdir(path):  # Projdi každou osobu v zadané složce
        persons = os.listdir(path + person)
        for known_person in persons:
            img = face_recognition.load_image_file(path + person + "/" + known_person)
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            faces = detector(rgb)  # dlib HOG detektor
            faces = [convert_and_trim_bb(img, r) for r in faces]  # převedení na opencv souřadnice
            encoding = face_recognition.face_encodings(rgb)[0]
            names = []
            name = "nerozpoznáno"
            if method == "MX":

                matches = face_recognition.compare_faces(training_data["encodings"], encoding)
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = training_data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)
            else:
                face_distance = face_recognition.face_distance(training_data["encodings"], encoding)
                idx = np.argsort(face_distance)[1:k]
                name = "neznámé"
                counts = {}
                for i in idx:
                    name = training_data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                #small_dist_index = np.argmin(face_distance)
                #name = training_data["names"][small_dist_index]
            if name in person:
                recognized += 1
                print("rozpoznáno: " + str(recognized))
            else:
                unrecognized += 1
                print("nerozpoznáno: " + str(unrecognized))
            if plot:
                for ((x, y, w, h), name) in zip(faces, names):
                    cv.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(rgb, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    cv2.imshow("Detekovaný obličej", rgb)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    end = time.time()
    print("Rozpoznání " + str(end - start))
    print("rozpoznáno: " + str(recognized))
    print("nerozpoznáno: " + str(unrecognized))


if __name__ == '__main__':
    recognized = 0
    unrecognized = 0

    # Přednastavení aplikace
    PATH = "Osobnost/dbs_faces/"
    DETECTOR = dlib.get_frontal_face_detector()
    METHOD = "NN"  # nn vzdálenost "NN" nebo max occur "MX"
    PLOT = False

    # main program
    training_data = get_encogings(PATH)  # load img and scrap only faces
    recognize(training_data, PATH, METHOD, DETECTOR, PLOT)
