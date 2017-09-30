from argparse import ArgumentParser
from cv2 import imread, cvtColor, COLOR_RGB2GRAY, rectangle, imshow, waitKey, destroyAllWindows
# from cv2 import imwrite
from wrappers.detect_face import FaceDetector
# This script allow you to detect a face based on OpenCV xml haarcascade file.
# You must need to change min_face_dim in the FaceDetector constructor.

default_casc_path = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"  # Path on fedora 25


def main():
    parser = ArgumentParser(description="This script allow you to detect a face based on OpenCV xml haarcascade file.\
     You must need to change min_face_dim in the FaceDetector constructor.")
    parser.add_argument("-haar", "--haarcascade-path", default=default_casc_path, dest="casc_path", help="Path to the haarcascade file you want to use.")
    parser.add_argument("-p", "--picture", required=True, dest="picture", help="Picture in which you look for faces")
    args = parser.parse_args()
    casc_path, picture = args.casc_path, args.picture

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
