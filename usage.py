from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import random


def check_final():

    # load the trained model to predict the images in the predict folder
    model = models.load_model('model.h5')

    PREDDIR = "./predict/"
    CATEGORIES = ["A", "B", "C", "D"]

    for category in CATEGORIES:
        path = os.path.join(PREDDIR, category)

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_GRAYSCALE)
            break
        break

    IMG_WIDTH = 60
    IMG_HEIGHT = 100
    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))

    predict_data = []
    print("Prediction Using Trained Model")

    def create_predict_data():
        for category in CATEGORIES:
            path = os.path.join(PREDDIR, category)
            class_num = CATEGORIES.index(category)

            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                    predict_data.append([new_array, class_num])
                except Exception:
                    pass

    create_predict_data()
    print(len(predict_data))

    # random.shuffle(predict_data)

    a = []  # feature set
    b = []  # labels

    for features, labels in predict_data:
        a.append(features)
        b.append(labels)

    a = np.array(a).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)

    y = model.predict(a)


check_final()
