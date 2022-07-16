import cv2 as cv
from bing_image_downloader import downloader
import matplotlib.pyplot as plt
import os


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
    face_cascade = cv.CascadeClassifier("data/viola_jones/haarcascade_frontalface_default.xml")
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 6)
    for (x, y, w, h) in faces:
        height = y + h  # konečné souřadnice
        width = x + w
        color = (255, 0, 0)
        stroke = 2
        img = cv.rectangle(img, (x, y), (width, height), color, stroke)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # plt.set_cmap('jet')  # set colormap
    # path1 = "Images\man1.jpg"
    # path2 = "Images\man3.jpg"
    # detectAndDisplay(cv.imread(path2))
    # detectAndDisplay(cv.imread(path1))
    person = "Databases/Angelina Jolie"
    downloader.download(person, limit=15, output_dir="Osobnost", adult_filter_off=True,
                        force_replace=False, timeout=60, verbose=True, filter="imagesize-large+filterui:face-face")
# images = load_images_from_folder("Osobnost/Johny Depp")
# show_detected_images(images)
