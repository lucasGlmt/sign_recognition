# PT4 - Reconnaissance de panneaux routier
Lucas GUILLEMINOT (2020 - 2021)


## Base de données
Les données proviennent de __Kaggle__ (https://www.kaggle.com/) qui est un site qui propose des données pour l'étude du Deep Learning ou du Machine Learning. La base que j'utilise à été conçue par __MyKola__ (https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
Cette base de données est parfaite, car elle contient 43 classes correspondant à 43 types de panneaux. De plus, elle fait partie __du domaine public__ donc elle est libre d'utilisation.

## Architecture
Pour ce projet, j'ai opté pour l'utilisation d'un perceptron multicouche. Voici son architecture : 
1 - Couche de convolution (fonction d'activation relu, filters = 32)
2 - Couche de convolution (fonction d'activation relu,  filters = 32)
3 - Couche de MaxPooling (pool_size de 2x2)
4 - Couche de Dropout
5 - Couche de convolution (fonction d'activation relu,  filters = 64)
6 - Couche de convolution (fonction d'activation relu,  filters = 64)
7 - Couche de MaxPooling (pool_size de 2x2)
8 - Couche de Dropout
9 - Vectorisation (flatten)
10 - Couche de Dense(taille 256, activation relu)
11 - Couche de dropout
12 - Couche de Dense (taille 43 (nombre de classes), activation softmax).

