# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:35:00 2020

@author: Zeng Fung

Code to run Feed-Forward Neural Network in Keras
"""

PERCENTAGE_OF_TRAIN = 0.8

import pandas as pd
import numpy as np
import data_augmentation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers #for l1 or l2 regularizers
from keras.callbacks import EarlyStopping #stop training when monitored argument stops decreasing/increasing

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

# delete variables that are no longer needed to allocate memory
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

# Setting up variables for neural network training
(NUM_OF_DATA, INPUT_NODES) = train_x.shape
OUTPUT_NODES = 2
LAYER1_NODES = 500
LAYER2_NODES = 500

EPOCHS = 300
BATCH_SIZE = 256
L1 = 0
L2 = 1e-4
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

# Training neural network

model = Sequential()
# first hidden layer
model.add(Dense(units = LAYER1_NODES, input_dim = INPUT_NODES,
                activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
# second hidden layer
model.add(Dense(units = LAYER2_NODES, activation = 'relu',
          kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
# output layer
model.add(Dense(units = OUTPUT_NODES, activation = 'softmax'))

# compile models with the learning rates set
opt = optimizers.adam(learning_rate = LEARNING_RATE)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

history = model.fit(nn_x_train, nn_y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, 
          validation_data = (nn_x_validation, nn_y_validation),
          callbacks = [EarlyStopping(monitor='val_accuracy', patience=20)]
          ) 

#evaluate model on test set
_, accuracy = model.evaluate(test_x, test_y_01)
print('Test Data Accuracy:', '{:.3f}'.format(accuracy))

# classify test data using the model
test_y_pred_01 = model.predict(test_x, verbose = 0)

test_y_pred = []
for i in range(len(test_y_pred_01)):
    if test_y_pred_01[i][0] > test_y_pred_01[i][1]:
        test_y_pred.append('Yes')
    else:
        test_y_pred.append('No')

# show confusion matrix 
confmat = confusion_matrix(test_y_true, test_y_pred, labels = ['Yes', 'No'])
print(confmat)