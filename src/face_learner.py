#!/usr/bin/env python3
# coding: utf-8
# This script allow you to learn a dataset, that you provide through a csv (see create_csv.py) and create a OpenCV xml file.
# Usage: python3 face_learner.py [path/to/detection/xml/haarcascade.xml] <path/to/csv/file.csv> <your_new_recognizer.xml>
from cv2.face import createLBPHFaceRecognizer  # can be change but dont forget to change into face_recognizer
from wrappers.faces_recognizer import Recognizer, get_dataset_csv
from sys import exit, argv, stderr


def main():
    if len(argv) in range(3, 5):
        if len(argv) == 3:
            casc_path = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25
            csv_file = argv[1]
            ouput_xml = argv[2]
        else:
            casc_path = argv[1]
            csv_file = argv[2]
            ouput_xml = argv[3]
    else:
        print("Usage: python3 face_learner.py [path/to/detection/xml/haarcascade.xml] <path/to/csv/file.csv> <your_new_recognizer.xml>", file=stderr)

    recognizer = Recognizer(createLBPHFaceRecognizer)
    recognizer.train(*get_dataset_csv(csv_file, casc_path))
    recognizer.save_recognizer(ouput_xml)


if __name__ == '__main__':
    main()
