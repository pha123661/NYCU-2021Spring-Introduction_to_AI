import os
import cv2
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    # Begin your code (Part 4)

    file = open(dataPath, "r")
    line = file.readline()

    while line:
        line = line.split()
        filename = line[0]
        face_num = int(line[1])
        face_set = []
        img = cv2.imread(os.path.join(os.path.dirname(dataPath), filename))
        for i in range(face_num):
            line = file.readline()
            line = line.split()
            line = [int(i) for i in line]
            x, y, w, h = line
            # crop, convert and resize image
            cropped_img = img[y:y+h, x:x+w]
            gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(
                gray_img, (19, 19), interpolation=cv2.INTER_NEAREST)

            # classify image
            if clf.classify(resized_img):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(img, (x, y),
                          (x+w, y+h), color, 3)
        cv2.imshow(filename, img)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)
        line = file.readline()

    file.close()

    # End your code (Part 4)
