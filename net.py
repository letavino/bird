import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import plot_model
from scipy.misc import imread
from keras.preprocessing.image import ImageDataGenerator
from keras import utils as np_utils

batch_size = 32
epochs = 500
img_size = (32,32)
input_shape = (img_size[0],img_size[1],3)
output = 2

print('\nload data')

with open("dataset_train.csv", "r") as f:
	train = list(np.array(list(csv.reader(f))).T)

with open("dataset_test.csv", "r") as f:
	test = np.array(list(csv.reader(f))).T

folder = 'mixed/'

xTrain = np.array([imread(folder+fname) for fname in train[0]]) / 255
yTrain = train[1]
xTest = np.array([imread(folder+fname) for fname in test[0]]) / 255
yTest = test[1]


integer_encoded = yTrain
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
yTrain = onehot_encoder.fit_transform(integer_encoded)

integer_encoded = yTest
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
yTest = onehot_encoder.fit_transform(integer_encoded)

datagen = ImageDataGenerator(
	rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    channel_shift_range=0.15,
    vertical_flip=True,
    rescale=None,
	preprocessing_function=None)

datagen.fit(xTrain)

print("\nBuild model")

model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5),activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(5, 5),activation='relu'))
model.add(Flatten())
model.add(Dense(5000, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(output, activation='softmax'))

print("\nStart training")

sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

history = model.fit_generator(datagen.flow(xTrain, yTrain, batch_size=batch_size), steps_per_epoch=len(xTrain) / 32, epochs=epochs, verbose=1,validation_data=(xTest,yTest))

score = model.evaluate(xTest, yTest, batch_size=batch_size)

print("\nTraining done")
model.save('model.h5')
print("\nModel saved")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()