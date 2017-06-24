from detect_face import FaceDetector
from sys import argv, exit, stderr
from PIL import Image
import cv2


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

    img = cv2.imread(picture)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detections = detector.detect(gray)

    for (x, y, w, h) in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Picture', img)

    #cv2.imwrite("detect.png", img)
    cv2.waitKey(0) & 0xFF == ord('q')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
