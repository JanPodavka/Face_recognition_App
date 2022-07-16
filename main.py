import cv2
import dlib
from kivy.clock import Clock
from kivy.core.image import Texture
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screen import Screen
from kivymd.uix.filemanager import MDFileManager
from bing_image_downloader import downloader
import cv2
from detection import hog, viola_jones, hsv  # IMPORT vlastních metod
from recognition import load_encodings, face_recogniton, load_mace, mace, load_pca, pca


class LoginWindow(Screen):
    pass


class DetectionWindow(Screen):
    def __init__(self, **kw):
        super().__init__()

    def on_pre_enter(self, *args):
        super().__init__()
        Clock.schedule_interval(self.update, 1.0)  # update 1 fps

    def update(self, dt):


        if self.ids['image'].source is None:
            return 0

        font = cv2.FONT_HERSHEY_DUPLEX
        app = MDApp.get_running_app()
        frame = cv2.imread(self.ids['image'].source)
        if frame is None:
            return 0
        # Nastavení zvolených detektorů
        if app.active_hog:
            frame = hog(frame, app.detector_hog, app.faces)
        if app.active_vj:
            frame = viola_jones(frame, app.detector_vj, app.faces)
        if app.active_hsv:
            frame = hsv(frame, app.faces)

        # rozpoznávání
        for (x, y, w, h, color) in app.faces:
            #cv2.rectangle(frame, (x, y), (w, h), color, 2)
            if app.active_face_rec and not app.active_pca and not app.active_mace:
                name = face_recogniton(app.encodings, frame, (x, y, w, h))
            if not app.active_face_rec and app.active_pca and not app.active_mace:
                name = pca(app.data_pca, frame, (x, y, w, h))
            if not app.active_face_rec and not app.active_pca and app.active_mace:
                name = mace(app.data_mace, frame, (x, y, w, h))
            if not app.active_face_rec and not app.active_pca and not app.active_mace:
                name = ""
            try:
                cv2.putText(frame, name, (x + 10, h - 10), font, 1.0, (255, 255, 255), 2)
            except:
                print("nikdo")


        app.faces = []
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids['image'].texture = image_texture

        ##Metody tlačítek

    def pressed_hog(self):
        app = MDApp.get_running_app()
        if app.active_hog:
            self.ids['hog'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['hog'].background_color = 0, 230, 0, 0.6

        app.active_hog = not app.active_hog

    def pressed_vj(self):
        app = MDApp.get_running_app()
        if app.active_vj:
            self.ids['vj'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['vj'].background_color = 0, 230, 0, 0.6
        app.active_vj = not app.active_vj

    def pressed_hsv(self):
        app = MDApp.get_running_app()
        if app.active_hsv:
            self.ids['hsv'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['hsv'].background_color = 0, 230, 0, 0.6
        app.active_hsv = not app.active_hsv

    def pressed_face_rec(self):
        app = MDApp.get_running_app()
        if app.active_face_rec:
            self.ids['face_rec'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['face_rec'].background_color = 0, 230, 0, 0.6

        app.active_face_rec = not app.active_face_rec

    def pressed_pca(self):
        app = MDApp.get_running_app()
        if app.active_pca:
            self.ids['pca'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['pca'].background_color = 0, 230, 0, 0.6
        app.active_pca = not app.active_pca

    def pressed_mace(self):
        app = MDApp.get_running_app()
        if app.active_mace:
            self.ids['mace'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['mace'].background_color = 0, 230, 0, 0.6
        app.active_mace = not app.active_mace


    def on_leave(self, *args):
        app = MDApp.get_running_app()
        app.active_mace = False
        app.active_pca = False
        app.active_vj = False
        app.active_hog = False
        app.active_face_rec = False
        app.active_hsv = False


class WelcomeWindow(Screen):
    pass


class CamWindow(Screen): ##### Reset když se přepne na jiné window
    def __init__(self, **kw):
        super().__init__()
        self.capture = None

    def on_pre_enter(self, *args):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30)  # update 30 fps

    def on_leave(self, *args):
        self.capture.release()
        app = MDApp.get_running_app()
        app.active_mace = False
        app.active_pca = False
        app.active_vj = False
        app.active_hog = False
        app.active_face_rec = False
        app.active_hsv = False

    # update every 1/30 sec
    def update(self, dt):


        ret, frame = self.capture.read()

        if ret:
            font = cv2.FONT_HERSHEY_DUPLEX
            app = MDApp.get_running_app()
            # Nastavení zvolených detektorů
            if app.active_hog:
                frame = hog(frame, app.detector_hog, app.faces)
            if app.active_vj:
                frame = viola_jones(frame, app.detector_vj, app.faces)
            if app.active_hsv:
                frame = hsv(frame, app.faces)

            # rozpoznávání
            for (x, y, w, h, color) in app.faces:
                #cv2.rectangle(frame, (x, y), (w, h), color, 2)
                if app.recognize_timer >= 120:
                    if app.active_face_rec and not app.active_pca and not app.active_mace:
                        name = face_recogniton(app.encodings, frame, (x, y, w, h))
                        cv2.putText(frame, name, (x + 6, h - 6), font, 1.0, (255, 255, 255), 1)
                    if not app.active_face_rec and app.active_pca and not app.active_mace:
                        name = pca(app.data_pca, frame, (x, y, w, h))
                        cv2.putText(frame, name, (x + 6, h - 6), font, 1.0, (255, 255, 255), 1)
                    if not app.active_face_rec and not app.active_pca and app.active_mace:
                        name = mace(app.data_mace, frame, (x, y, w, h))
                        print(name)
                        cv2.putText(frame, name, (x + 6, h - 6), font, 1.0, (255, 255, 255), 1)
                    if not app.active_face_rec and not app.active_pca and not app.active_mace:
                        name = ""

                if app.recognize_timer > 120:
                    app.recognize_timer = 0 # reset timeru# reset timeru
                else:
                    app.recognize_timer += 1


            app.faces = []
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids['image'].texture = image_texture

    ##Metody tlačítek

    def pressed_hog(self):
        app = MDApp.get_running_app()
        if app.active_hog:
            self.ids['hog'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['hog'].background_color = 0, 230, 0, 0.6

        app.active_hog = not app.active_hog

    def pressed_vj(self):
        app = MDApp.get_running_app()
        if app.active_vj:
            self.ids['vj'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['vj'].background_color = 0, 230, 0, 0.6
        app.active_vj = not app.active_vj

    def pressed_hsv(self):
        app = MDApp.get_running_app()
        if app.active_hsv:
            self.ids['hsv'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['hsv'].background_color = 0, 230, 0, 0.6
        app.active_hsv = not app.active_hsv


    def pressed_face_rec(self):
        app = MDApp.get_running_app()
        if app.active_face_rec:
            self.ids['face_rec'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['face_rec'].background_color = 0, 230, 0, 0.6

        app.active_face_rec = not app.active_face_rec

    def pressed_pca(self):
        app = MDApp.get_running_app()
        if app.active_pca:
            self.ids['pca'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['pca'].background_color = 0, 230, 0, 0.6
        app.active_pca = not app.active_pca

    def pressed_mace(self):
        app = MDApp.get_running_app()
        if app.active_mace:
            self.ids['mace'].background_color = 0, 0, 255, 0.6
        else:
            self.ids['mace'].background_color = 0, 230, 0, 0.6
        app.active_mace = not app.active_mace



class AddfaceWindow(Screen):

    def download_person(self):
        person = self.ids['find_per'].text
        if person is None or person == "": #is
            return 0

        downloader.download(person, limit=9, output_dir="Osobnost", adult_filter_off=True,
                            force_replace=False, timeout=60, verbose=True)
        for i in range(1, 10):
            self.ids['o' + str(i)].source = "Osobnost/" + person + "/" + "Image_" + str(i) + ".jpg"


class WindowManager(ScreenManager):
    pass


class MyApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #  Detektory
        self.detector_vj = cv2.CascadeClassifier("data/viola_jones/haarcascade_frontalface_default.xml")
        self.detector_hog = dlib.get_frontal_face_detector()
        #  Aktivace detektorů
        self.active_hog = False
        self.active_vj = False
        self.active_hsv = False
        # aktivace rozpoznávání
        self.active_face_rec = False
        self.active_pca = False
        self.active_mace = False
        # list souřadnic
        self.faces = []
        # data
        self.names = []
        self.encodings = load_encodings()
        self.data_mace = load_mace()
        self.data_pca = load_pca()
        # timer
        self.recognize_timer = 70
        #  FileManager

        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            preview=True,
        )

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        screen = Builder.load_file("styly.kv")
        return screen

    #  File Manager manipulating
    def file_manager_open(self):
        self.file_manager.show('/')

    def select_path(self, path):
        self.exit_manager(path)
        self.root.get_screen('detection').ids.image.source = path
        toast(path)

    def exit_manager(self, path, *args):
        self.manager_open = False
        self.file_manager.close()
        return path


if __name__ == '__main__':
    MyApp().run()
