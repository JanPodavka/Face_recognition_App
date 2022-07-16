import cv2 as cv
from bing_image_downloader import downloader
import matplotlib.pyplot as plt
import os

import dlib


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


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def show_detected_images(images):
    for img in images:
        detectAndDisplay(img)


def detectAndDisplay(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img)
    boxes = [convert_and_trim_bb(img, r) for r in faces]
    for (x, y, w, h) in boxes:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    plt.set_cmap('jet')  # set colormap
    path1 = "Images\man1.jpg"
    path2 = "Images\man3.jpg"
    detectAndDisplay(cv.imread(path2))
    detectAndDisplay(cv.imread(path1))
    person = "celebrity"
    # downloader.download(person, limit=100, output_dir="Osobnost", adult_filter_off=True,
    #                      force_replace=False, timeout=60, verbose=True)
    images = load_images_from_folder("Osobnost/celebrity")
    show_detected_images(images)

