# -*- coding: utf-8 -*-
"""ear_biometrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-HlByiHEYqT6oLyDAbk9FMYy3NbCUIKS

# Ear Biometrics System
"""

"""## Creating Model

### Importing Libraries
"""


def train():
    from __future__ import print_function

    import os
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras.utils import to_categorical
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import random

    """### Loading the data"""

    DATADIR = './train/'
    CATEGORIES = ['A', 'B', 'C', 'D']

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_GRAYSCALE)
            break
        break

    IMG_WIDTH = 60
    IMG_HEIGHT = 100
    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))

    training_data = []

    def create_training_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                    training_data.append([new_array, class_num])
                except Exception:
                    pass

    create_training_data()
    print(len(training_data))

    random.shuffle(training_data)

    a = []
    b = []

    for features, labels in training_data:
        a.append(features)
        b.append(labels)

    a = np.array(a).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    print(a.shape)

    X_train = a/255.0
    y_train = b

    TESTDIR = './test/'

    testing_data = []

    def create_testing_data():
        for category in CATEGORIES:
            path = os.path.join(TESTDIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                    testing_data.append([new_array, class_num])
                except Exception as e:
                    pass

    create_testing_data()
    print(len(testing_data))

    random.shuffle(testing_data)

    p = []
    q = []

    for features, labels in testing_data:
        p.append(features)
        q.append(labels)

    p = np.array(p).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    print(p.shape)

    X_test = p/255.0
    y_test = q

    """### Creating the ConvNet"""

    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu',
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        keras.layers.Conv2D(64, 3, strides=2, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, strides=2, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, to_categorical(y_train), batch_size=32,
              epochs=5, validation_data=((X_test, to_categorical(y_test))))

    scores = model.evaluate(X_test, to_categorical(y_test))

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

    model.save('model.h5')


train()
