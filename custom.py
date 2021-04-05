## Import des librairies
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

def showImages(arr):
     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
     axes = axes.flatten()
     for img, ax in zip(arr, axes):
         ax.imshow(img)
     plt.tight_layout()
     plt.show()

## Dossiers
train_dir = './data/train'
validate_dir = './data/validate'
test_dir = './data/test'

## Training folders
train_stop_dir = os.path.join(train_dir, 'stop')
train_yeild_dir = os.path.join(train_dir, 'yield')
train_crosswalk_dir = os.path.join(train_dir, 'crosswalk')


## Validation folders
validate_stop_dir = os.path.join(validate_dir, 'stop')
validate_yeild_dir = os.path.join(validate_dir, 'yield')
validate_crosswalk_dir = os.path.join(validate_dir, 'crosswalk')

## Tests folder
test_signs_folder = os.path.join(test_dir, 'signs')


num_stop_train = len(os.listdir(train_stop_dir))
num_yield_train = len(os.listdir(train_yeild_dir))
num_stop_validate = len(os.listdir(validate_stop_dir))
num_yield_validate = len(os.listdir(validate_yeild_dir))

train_total = num_stop_train + num_yield_train
validate_total = num_stop_validate + num_yield_validate
test_total = len(os.listdir(test_signs_folder))


batch_size = 32
img_size = 150

img_gen = ImageDataGenerator(rescale=1./255);


train_img_gen = img_gen.flow_from_directory(batch_size=batch_size,
                                           directory=train_dir,
                                           shuffle=True,
                                           target_size=(img_size, img_size),
                                           class_mode='binary')

validate_img_gen = img_gen.flow_from_directory(batch_size=batch_size,
                                               directory=validate_dir,
                                               shuffle=False,
                                               target_size=(img_size, img_size),
                                               class_mode='binary')

test_img_gen = img_gen.flow_from_directory(batch_size=batch_size,
                                               directory=test_dir,
                                               shuffle=False,
                                               target_size=(img_size, img_size),
                                               class_mode=None)


## Perceptron multicouche

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


fit_result = model.fit_generator(
            train_img_gen,
            steps_per_epoch=int(np.ceil(train_total / float(batch_size))),
            epochs=5,
            validation_data=validate_img_gen,
            validation_steps=int(np.ceil(validate_total / float(batch_size)))
            )




model.save_weights('model.h5')






