import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#seed random number generator
keras.utils.set_random_seed(42)

#Download image data directly from Keras
(x_train, y_train), (x_test,y_test) = keras.datasets.fashion_mnist.load_data()

#create a python list for labels
labels = ["T-shirt/top",
          "Trouser",
          "Pullover",
          "Dress",
          "Coat",
          "Sandal",
          "Shirt",
          "Sneaker",
          "Bag",
          "Ankle boot"]



#normalize data
x_train = x_train / 255
x_test = x_test / 255

x_train.shape

#Double check and confirm number of images match what is expected
print(f"Number of images and size dimension: {x_train.shape}")

#Expand the dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train.shape
print(f"{x_train.shape}")



# **************Define the MOdel***************

#input layer
input = keras.Input(shape=x_train.shape[1:])

#First Convolutional Block

#convolutional layer
x = keras.layers.Conv2D(32,                     # Number of filters
                    kernel_size = (2, 2),       #The shape of each filter
                    activation = "relu",        #RELU activation
                    name = "Conv_1")(input)

#Pooling layer
x = keras.layers.MaxPool2D()(x)

#Second Convolutional Block

#Convolutional layer
x = keras.layers.Conv2D(32,                     # Number of filters
                    kernel_size = (2, 2),       #The shape of each filter
                    activation = "relu",        #RELU activation
                    name = "Conv_2")(x)

#Pooling layer
x = keras.layers.MaxPool2D()(x)


#Flatten layer
x = keras.layers.Flatten()(x)

#Fully connected dense layer
x = keras.layers.Dense(256, activation = "relu")(x)


#output Softmax
output = keras.layers.Dense(10, activation = "softmax")(x)

model = keras.Model(input, output)

model.summary()


#Set optimization parameters
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = "adam",
              metrics = ["accuracy"])


#Train model
history = model.fit(x_train,
                      y_train,
                      batch_size = 64,
                      epochs = 10,
                      validation_split = 0.2)


#Evaluate model
score = model.evaluate(x_test, y_test)
print("Test Accuracy:", score[1])

