# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:35:00 2020

@author: Zeng Fung

Code to run Convolutional Neural Network in Keras
"""

PERCENTAGE_OF_TRAIN = 0.8

import pandas as pd
import numpy as np
import data_augmentation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Read file and split data into training and test data
# import mask.csv
full_data = pd.read_csv('mask.csv', header = 0)

# split into two separate data frames by images with and without masks
with_mask = full_data[full_data.with_mask == 'Yes']
without_mask = full_data[full_data.with_mask == 'No']

(train_x_withmask, test_x_withmask, train_y_withmask, test_y_withmask) = train_test_split(with_mask.iloc[:,1:], with_mask.iloc[:,0], train_size = 0.8, test_size = 0.2, random_state = 1)
(train_x_withoutmask, test_x_withoutmask, train_y_withoutmask, test_y_withoutmask) = train_test_split(without_mask.iloc[:,1:], without_mask.iloc[:,0], train_size = 0.8, test_size = 0.2, random_state = 1)

train_x = np.vstack((train_x_withmask, train_x_withoutmask))
train_y = pd.concat([train_y_withmask, train_y_withoutmask], axis = 0).reset_index(drop = True)
train_y = np.array(train_y)

test_x = np.vstack((test_x_withmask, test_x_withoutmask))
test_y_true = pd.concat([test_y_withmask, test_y_withoutmask], axis = 0).reset_index(drop = True)
test_y_true = np.array(test_y_true)

# delete variables that are no longer needed
del with_mask
del without_mask
del train_x_withmask
del train_x_withoutmask
del test_x_withmask
del test_x_withoutmask
del train_y_withmask
del train_y_withoutmask
del test_y_withmask
del test_y_withoutmask

# Setting up variables for Neural Network
INPUT_NODES = train_x.shape[1]
OUTPUT_NODES = 2

EPOCHS = 30
BATCH_SIZE = 256
L1 = 0
L2 = 1e-2
LEARNING_RATE = 1e-4

# splitting training data into train and validation data sets
(nn_x_train, nn_x_validation, nn_y_train, nn_y_validation) = train_test_split(train_x, train_y, train_size = 0.8, test_size = 0.2, random_state = 1)

# increase number of training data
(nn_x_train, nn_y_train) = data_augmentation.increase_data(nn_x_train, nn_y_train, (128,128,3))

# convert test and training y values from 'Yes' or 'No' to [1,0] or [0,1] respectively
onehot_encoder = OneHotEncoder(sparse=False)
nn_y_train = nn_y_train.reshape(len(nn_y_train),1)
nn_y_train = onehot_encoder.fit_transform(nn_y_train)
nn_y_validation = nn_y_validation.reshape(len(nn_y_validation),1)
nn_y_validation = onehot_encoder.fit_transform(nn_y_validation)
test_y_01 = test_y_true.reshape(len(test_y_true),1)
test_y_01 = onehot_encoder.fit_transform(test_y_01)

# changing input shape of training, validation, and test data
nn_x_train = nn_x_train.reshape((len(nn_x_train), 128, 128, 3))
nn_x_validation = nn_x_validation.reshape((len(nn_x_validation), 128, 128, 3))
test_x = test_x.reshape((len(test_x), 128, 128, 3))

# build 2d-convolutional neural network
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
from keras import regularizers #for l1 or l2 regularizers
from keras.callbacks import EarlyStopping #stop training when monitored argument stops decreasing/increasing

model = Sequential()
# adding first 2d-convolutional layer
model.add(Conv2D(32, kernel_size = (3,3), input_shape = (128,128,3), padding = "same",
                 kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
model.add(Activation("relu"))
model.add(BatchNormalization())
# adding pooling layer
model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))
# adding second 2d-convolutional layer
model.add(Conv2D(64, kernel_size = (3,3), padding = "same",
                  kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
model.add(Activation("relu"))
model.add(BatchNormalization())
# adding pooling layer
model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))
# adding third 2d-convolutional layer
model.add(Conv2D(128, kernel_size = (3,3), padding = "same",
                  kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
model.add(Activation("relu"))
model.add(BatchNormalization())
# adding pooling layer
model.add(MaxPooling2D(pool_size = (2,2), padding = "valid"))
# flatten convolutional layers, move to dense layer
model.add(Flatten())
# adding first dense layer
model.add(Dense(units = 64, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
model.add(Dropout(0.2))
# adding first dense layer
model.add(Dense(units = 32, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
model.add(Dropout(0.2))
# adding output layer
model.add(Dense(units = OUTPUT_NODES, activation = "softmax"))

# compile models with the learning rates set
opt = optimizers.adam(learning_rate = LEARNING_RATE)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

history = model.fit(nn_x_train, nn_y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, 
          validation_data = (nn_x_validation, nn_y_validation),
          callbacks = [EarlyStopping(monitor='val_loss', patience=20)]
          ) 

#evaluate model on test set
_, accuracy = model.evaluate(test_x, test_y_01)

print('Test Data Accuracy:', '{:.3f}'.format(accuracy))

# plot cost and accuracy of training and validation over epoch
train_cost = history.history['loss']
val_cost = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
ep = list(range(1, len(train_acc)+1))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize = (40,20))
ax[0].plot(ep, train_cost, 'ro-', label = 'Train')
ax[0].plot(ep, val_cost, 'bo-', label = 'Val')
ax[0].set_xlabel('EPOCH', fontsize = 20)
ax[0].set_ylabel('cost', fontsize = 20)
ax[0].set_title('Change in cost', fontsize = 25)
ax[0].legend(fontsize = 20)

ax[1].plot(ep, train_acc, 'ro-', label = 'Train')
ax[1].plot(ep, val_acc, 'bo-', label = 'Val')
ax[1].plot(ep, [accuracy]*len(ep), 'k-')
ax[1].set_xlabel('EPOCH', fontsize = 20)
ax[1].set_ylabel('accuracy', fontsize = 20)
ax[1].set_title('Change in accuracy', fontsize = 25)
ax[1].legend(fontsize = 20)
plt.show()