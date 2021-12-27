import os
import numpy as np
import cv2


def loadImages(dataPath):
    # Begin your code (Part 1)

    dataset = []
    face_img = [os.path.join(dataPath, "face", f)
                for f in os.listdir(os.path.join(dataPath, "face"))]
    non_face_img = [os.path.join(dataPath, "non-face", f)
                    for f in os.listdir(os.path.join(dataPath, "non-face"))]
    flag = 1
    for i in face_img:
        img = cv2.imread(i, -1)
        tmp = (img, flag)
        dataset.append(tmp)

    flag = 0
    for i in non_face_img:
        img = cv2.imread(i, -1)
        tmp = (img, flag)
        dataset.append(tmp)

    # End your code (Part 1)
    return dataset
