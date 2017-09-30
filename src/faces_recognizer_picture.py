from argparse import ArgumentParser
from sys import stderr
import cv2
from cv2.face import LBPHFaceRecognizer_create
from cv2 import imshow, waitKey, destroyAllWindows, VideoCapture, cvtColor, rectangle
from cv2 import COLOR_BGR2GRAY
import numpy as np
from csv import reader as csvreader
from wrappers.detect_face import FaceDetector
from wrappers.faces_recognizer import Recognizer, get_dataset_csv
# This script allow you to recognize a face based on a xml file (see face_learner.py) or a csv (see create_csv.py)


default_casc_path = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25


def main():
    parser = ArgumentParser(description="This script allow you to recognize a face based on a xml file (see face_learner.py) or a csv (see create_csv.py)")
    parser.add_argument("-haar", "--haarcascade-path", default=default_casc_path, dest="casc_path", help="Path to the haarcascade file you want to use.")
    parser.add_argument("-r", "--recognizer_file", required=True, dest="recognizer_file", help="Recognizer file in format csv or xml")
    parser.add_argument("-p", "--picture", required=True, dest="picture", help="Picture in which you look for faces")
    args = parser.parse_args()
    casc_path, recognizer_file, picture = args.casc_path, args.recognizer_file, args.picture

    recognizer = Recognizer(LBPHFaceRecognizer_create)
    if recognizer_file.endswith("xml"):
        recognizer.set_recognizer_xml(recognizer_file)
    elif recognizer_file.endswith("csv"):
        recognizer.train(*get_dataset_csv(recognizer_file, casc_path, min_face_dim=(100, 100)))
    else:
        print("Your file do not match a valid type", file=stderr)
        exit(1)

    detector = FaceDetector(casc_path, min_face_dim=(100, 100))

    img = cv2.imread(picture)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detections = detector.detect(gray)

    for (x, y, w, h) in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        nbr_predicted = recognizer.predict(gray[y: y + h, x: x + w])
        if nbr_predicted is not None:
            nb = nbr_predicted[0]
            txt = "Subject #{}".format(nb)
            cv2.putText(img, txt, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Picture', img)
    cv2.waitKey(0) & 0xFF == ord('q')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
