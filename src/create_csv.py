#!/usr/bin/env python3
# coding: utf-8
# script based on http://docs.opencv.org/2.4/modules/contrib/doc/facerec/tutorial/facerec_video_recognition.html
from sys import exit, stderr, argv
import os.path
from re import match
# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  koisell@fedora:~/faces/
#  .
#  |-- README
#  |-- s-1_bob
#  |-- s1
#  |   |-- 1.png
#  |   |-- ...
#  |   |-- 10.jpg
#  |-- s2_alice
#  |   |-- 1.png
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
# This script match directories named s(integer)AnythingYouWant
# integer will be the class associated with pictures found in the directory.
# Your picture must be followed by the extension png, jpg, pgm or jpeg (not case sensitive)

if __name__ == "__main__":

    if len(argv) != 2:
        print("usage: create_csv <base_path> > file.csv")
        exit(1)

    BASE_PATH = argv[1]
    SEPARATOR = "|"

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            number = match("s(-?\d+).*", subdirname)
            if number:
                label = int(number.group(1))
                for filename in os.listdir(subject_path):
                    if match(".*\.(?:png|jpg|pgm|jpeg)", filename.lower()) is not None:
                        abs_path = "%s/%s" % (os.path.abspath(subject_path), filename)
                        print("%s%s%d" % (abs_path, SEPARATOR, label))
                label += 1
