# Aplikace

spuštění přes skript main.py

Aplikace využívá 3 metody detekce:

- Viola - Jones (HAAR) - knihovna openCV
- Histogram orientovaných gradientů (HOG) - knihovna DLIB
- Segmentace v barevném prostoru HSV

A 3 metody identifikace:

- PCA
- Návržené filtry MACE
- Deep matric learning - knihovna face_recognition

<img src="https://github.com/JanPodavka/Face_recognition_App/blob/main/Images/porovn%C3%A1n%C3%AD%20detekce.png">

Samotná aplikace umožňuje funkcionality:

- Testování metod na libovolném obrázku:

<img src="https://github.com/JanPodavka/Face_recognition_App/blob/main/Images/1.png">

- Testování metod na videokameře

- Stažení 9 fotografií osob dle zadaného jména:

<img src="https://github.com/JanPodavka/Face_recognition_App/blob/main/Images/2.png">


## testing.py

- Soubor pro testování detekovacích metod
- Prozatím detekce obličeje pomocí Hue colorspace

## Viola_jones.py

- Pro špatný dataset zatížený šumem (zakrytí obličeje, žadný obličej, různé úhly, špatná kvalita) -> 54%
- Pro dataset obličejů -> 88%

## hog.py

- Pro špatný dataset zatížený šumem (zakrytí obličeje, žadný obličej, různé úhly, špatná kvalita) -> 64 %
- Pro dataset obličejů -> 93%

## face_rec.py

Aplikace pro testování úspěšnosti rozpoznávání na přiloženém datasetu

## detection.py

- slouží pro vizuální porovnání předchozích 2 metod na libovolném datasetu

## Script pro tvorbu datasetu s filtry pro vyhledávání

python ./bbid.py -s "name" -o "output_file" --filters +filterui:face-face+filterui:imagesize-large --limit 20

## Trénovací dataset

Dataset obsahující 50 osob po 15 fotografiích v souboru data/data_train.pkl ve formátu data = {"encodings": encodings, "names": labels}

- Adam_Sandler
- Angelina_Jolie
- Anne_Hathaway
- Ariana_Grande
- Arnold_Schwarzenegger
- Brad_Pitt
- Bruce_Willis
- Cristiano_Ronaldo
- Daniel_Radcliffe
- Dwayne_Johnson
- Ed_Sheeran
- Ema_Watson
- Emily_Blunt
- Harrison_Ford
- Harry_Styles
- Helena_Bonham_Carter
- Christian_Bale
- Jack_Nicholson
- Jackie_Chan
- Jennifer_Aniston
- Jennifer_Lawrence
- Jennifer_Lopez
- Jessica_Alba
- Jim_Carrey
- Kate_Winslet
- Keira_Knightley
- Kevin_Spacey
- Kim_Kardashian
- Kylie_Jenner
- Leonardo_DiCaprio
- Lionel_Messi
- Megan_Fox
- Morgan_Freeman
- Meryl_Streep
- Natalie_Portman
- Nicolas_Cage
- Nicole_Kidman
- Penelope_Cruz
- Robert_de_Niro
- Robert_Downey_jr
- Robert_Pattinson
- Sandra_Bullock
- Scarlett_Johansson
- Sylvester_Stallone
- Timothee_Chalamet
- Tom_Cruise
- Tom_Hanks
- Tom_Hardy
- Tom_Holland
- Will_Smith

## Seznam použitých knihoven

- Matplotlib
- opencv
- Kivy (GUI)
- KivyMD (designs)
- bing_image_downloader
- mahotas (CoM calculate)
- numpy
- face_recogniton
- dlib

