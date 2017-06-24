#!/usr/bin/env python3
# coding: utf-8

from cv2.face import createLBPHFaceRecognizer  # can be change but dont forget to change into face_recognizer
from faces_recognizer import Recognizer
from sys import exit, argv


def main():
    if len(argv) < 1:
        exit(1)

    if len(argv) > 2:
        cascPath = argv[2]
    else:
        cascPath = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25

    recognizer = Recognizer(createLBPHFaceRecognizer)
    recognizer.train(*recognizer.get_dataset_csv("faces_dataset.csv", cascPath))
    print(argv[1])
    recognizer.save_recognizer(argv[1])


if __name__ == '__main__':
    main()
