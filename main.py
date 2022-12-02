import tensorflow as tf
import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt
#import cv2 as cv
import copy

#import csv file
filePath = input("Type relative path to training data:\t")

dataset_file = pd.read_csv(str(filePath))

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#converts csv to numpy array
dataset = dataset_file.to_numpy()

#labels array
y_train = np.zeros(len(dataset))
#image data array
x_train = []

for i in range(0, len(dataset)):
    #first column for each row is the letter with a corresponding integer
    y_train[i] = dataset[i][0]
    #subsequent columns have 0-255 color values representing each pixel 
    x_train.append(dataset[i][1:len(dataset)])

#converts list to numpy array
x_train = np.array(x_train)

#converts 0-255 color value to 0.0-1.0 color value
x_train = tf.keras.utils.normalize(x_train)

#copies first image for displaying in matplot
showImage = copy.deepcopy(x_train[0])
#finds image dimensions
imgDimensions = math.sqrt(len(showImage))
#converts it to a matrix for matplot
showImage = np.reshape(showImage, (imgDimensions, imgDimensions))
#displays as grayscale image
plt.imshow(showImage, cmap='binary')
plt.show()


#number of hidden layers for model
numHiddenLayers = input("Enter number of hidden layers:\t")
#number of perceptrons per hidden layer
numPerceps = input("Enter number of perceptrons per hidden layer:\t")

#makes the model a simple sequential model
model = tf.keras.Sequential()
#since data is already a 1d array just make the input layer the image width by height
model.add(tf.keras.layers.Input(shape=(28*28,)))
for i in range(0, int(numHiddenLayers)):
    #creates hidden layers
    model.add(tf.keras.layers.Dense(int(numPerceps), activation=tf.nn.relu))
#creates output layer, 26 for each letter of the alphabet
model.add(tf.keras.layers.Dense(26, activation=tf.nn.softmax))

#number of epochs for training
numEpochs = input("Number of epochs:\t")

#compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#train data
model.fit(x_train, y_train, epochs=50)


