## Importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
import tqdm


## Donnés
data = []
labels = []
nbClasses = 43

## Importation des données
print("Importation des données")
for i in range (nbClasses):
    print("-- Import de la classe", i, "/", nbClasses)
    path = os.path.join(os.getcwd(), 'train',str(i))
    images = os.listdir(path)
    imgCount = 0
    for j in images:
        print(imgCount, "/", len(images), "(", (imgCount/(len(images))*100), "% )  -- Classe", i, "/", nbClasses)
        try:
            image = Image.open(path + '\\'+ j)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
            imgCount += 1
        except:
            print("Error loading image")
            

## Convertir les images en tableaux de bytes
data = np.array(data)
labels = np.array(labels)


print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=68)

y_train = to_categorical(y_train, nbClasses)
y_test = to_categorical(y_test, nbClasses)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



batch_size = 32
img_size = 150

img_gen = ImageDataGenerator(rescale=1./255)

train_img_gen = img_gen.flow_from_directory(batch_size=batch_size,
                                           directory=X_train,
                                           shuffle=True,
                                           target_size=(img_size, img_size),
                                           class_mode='binary')

test_img_gen = img_gen.flow_from_directory(batch_size=batch_size,
                                               directory=X_test,
                                               shuffle=False,
                                               target_size=(img_size, img_size),
                                               class_mode=None)

                                               



## Architecture du perceptron multicouche
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(nbClasses, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


## Entrainer le model

#history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
history = model.fit_generator(train_img_gen,
            steps_per_epoch=int(np.ceil(train_total / float(batch_size))),
            epochs=12,
            validation_data=validate_img_gen,
            validation_steps=int(np.ceil(validate_total / float(batch_size))))


## Sauvegarder le model
model.save("model.h5")


## Affichage des figures
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


plt.figure(1)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


from sklearn.metrics import accuracy_score
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test=np.array(data)

pred = model.predict_classes(X_test)
## Afficher la précision
print(accuracy_score(labels, pred))