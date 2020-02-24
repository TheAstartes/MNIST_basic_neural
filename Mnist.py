import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import  to_categorical
import random

np.random.seed(0)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape[0])

assert (x_train.shape[0] == y_train.shape[0]), "Num of imgs != num of labels"
assert (x_test.shape[0] == y_test.shape[0]),  "Num of imgs != num of labels"

assert(x_train.shape[1:] == (28,28)), "Wrong dimensions"
assert(x_test.shape[1:] == (28,28)), "Wrong dimensions"

num_of_samples = []

cols = 5
num_of_classes = 10

""""
fig, axs = plt.subplots(nrows=num_of_classes, ncols=cols, figsize=(5 , 10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_of_classes):
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
"""
print(num_of_samples)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train = x_train/255
x_test = x_test/255

num_pixels = 784

x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim= num_pixels, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())

history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size= 200, verbose=1, shuffle= 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('epoch')

import requests
from PIL import Image
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'

response = requests.get(url, stream=True)
img = Image.open(response.raw)

import cv2

img_array = np.asanyarray(img)
resized = cv2.resize(img_array, (28,28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)

image = image/255
image = image.reshape(1, 784)

prediction = model.predict_classes(image)
print("predicted digit:", str(prediction))

plt.show()
