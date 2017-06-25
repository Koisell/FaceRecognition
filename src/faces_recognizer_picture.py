from sys import argv, exit, stderr
import cv2
from cv2.face import createLBPHFaceRecognizer
from cv2 import imshow, waitKey, destroyAllWindows, VideoCapture, cvtColor, rectangle
from cv2 import COLOR_BGR2GRAY
import numpy as np
from csv import reader as csvreader
from wrappers.detect_face import FaceDetector
from wrappers.faces_recognizer import Recognizer, get_dataset_csv


def main():
    if len(argv) == 3:
        casc_path = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25
        recognizer_file = argv[1]
        picture = argv[2]
    elif len(argv) == 4:
        casc_path = argv[1]
        recognizer_file = argv[2]
        picture = argv[3]
    else:
        print("You must provide a picture", file=stderr)
        print("Usage: python3 detect_face_picture.py [path to recognizer.xml] <path to csv|path to xml> <path to picture>", file=stderr)
        exit(1)

    recognizer = Recognizer(createLBPHFaceRecognizer)
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
