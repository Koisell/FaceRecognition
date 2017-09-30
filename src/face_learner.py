#!/usr/bin/env python3
# coding: utf-8
# This script allow you to learn a dataset, that you provide through a csv (see create_csv.py) and create a OpenCV xml file.
from argparse import ArgumentParser
from cv2.face import LBPHFaceRecognizer_create  # can be change but dont forget to change into face_recognizer
from wrappers.faces_recognizer import Recognizer, get_dataset_csv

default_casc_path = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25


def main():
    parser = ArgumentParser(description="This script allow you to learn a dataset, that you provide through a csv (see create_csv.py) and create a OpenCV xml file.")
    parser.add_argument("-haar", "--haarcascade-path", default=default_casc_path, dest="casc_path", help="Path to the haarcascade file you want to use.")
    parser.add_argument("-i", "--csv", required=True, dest="csv_file", help="CSV file in format sep=| and new row=\\n (see create_csv.py)")
    parser.add_argument("-o", "--xml", required=True, dest="ouput_xml", help="File which will store the faces model.")
    args = parser.parse_args()
    casc_path, csv_file, ouput_xml = args.casc_path, args.csv_file, args.ouput_xml

    recognizer = Recognizer(LBPHFaceRecognizer_create)
    recognizer.train(*get_dataset_csv(csv_file, casc_path))
    recognizer.save_recognizer(ouput_xml)


if __name__ == '__main__':
    main()
