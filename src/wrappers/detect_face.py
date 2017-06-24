#!/usr/bin/env python3
# coding: utf-8

import cv2
import sys


class FaceDetector():
    def __init__(self, casc_path, min_face_dim=(50, 50), min_neighbors=5, scale_factor=1.1):
        self.casc_path = casc_path
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        self.min_face_dim = min_face_dim
        self.face_cascade = cv2.CascadeClassifier(self.casc_path)

    def detect(self, frame):
        return self.face_cascade.detectMultiScale(
            frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_dim
        )

    def set_casc_path(self, new_casc_path):
        self.casc_path = new_casc_path
        self.face_cascade = cv2.CascadeClassifier(self.cascPath)


def main():
    if len(sys.argv) > 1:
        cascPath = sys.argv[1]
    else:
        cascPath = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25

    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)
    min_face_dim = (50, 50)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_dim)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
