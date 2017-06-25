from sys import argv, exit, stderr
from cv2 import imread, cvtColor, COLOR_RGB2GRAY, rectangle, imshow, waitKey, destroyAllWindows
# from cv2 import imwrite
from wrappers.detect_face import FaceDetector
# This script allow you to detect a face based on OpenCV xml haarcascade file.
# You must need to change min_face_dim in the FaceDetector constructor.
# Usage: python3 detect_face_picture.py [path to recognizer.xml] <path to picture>


def main():
    if len(argv) == 2:
        casc_path = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25
        picture = argv[1]
    elif len(argv) == 3:
        casc_path = argv[1]
        picture = argv[2]
    else:
        print("You must provide a picture", file=stderr)
        print("Usage: python3 detect_face_picture.py [path to recognizer.xml] <path to picture>", file=stderr)
        exit(1)

    detector = FaceDetector(casc_path, min_face_dim=(100, 100))

    img = imread(picture)
    gray = cvtColor(img, COLOR_RGB2GRAY)
    detections = detector.detect(gray)

    for (x, y, w, h) in detections:
        rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    imshow('Picture', img)

    # cv2.imwrite("detect.png", img)
    waitKey(0) & 0xFF == ord('q')
    destroyAllWindows()


if __name__ == '__main__':
    main()
