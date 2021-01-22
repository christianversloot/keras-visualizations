##
## Read corresponding blog article at:
## https://www.machinecurve.com/index.php/2019/10/07/how-to-visualize-a-model-with-keras
##

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Set input shape
sample_shape = input_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# Number of classes
no_classes = 10

# Reshape data 
input_train = input_train.reshape(len(input_train), input_shape[0], input_shape[1], input_shape[2])
input_test  = input_test.reshape(len(input_test), input_shape[0], input_shape[1], input_shape[2])

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Vertical plot
plot_model(model, to_file='model.png')

# Horizontal plot
plot_model(model, to_file='model_horizontal.png', rankdir='LR')