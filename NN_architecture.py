# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:37:10 2020

@author: Zeng Fung

Code to experiment and compare the effects of different Neural Network 
architectures in Keras
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
(NUM_OF_DATA, INPUT_NODES) = train_x.shape
OUTPUT_NODES = 2

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

from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers #for l1 or l2 regularizers
import matplotlib.pyplot as plt

# prepare dataset with input and output scalers, can be none
def transform_dataset(input_scaler, train_X, val_X, test_X):
	# scale inputs
    if input_scaler is not None:
		# fit scaler
        input_scaler.fit(train_X)
		# transform training dataset
        trainX = input_scaler.transform(train_X)
        # transform validation dataset
        validX = input_scaler.transform(val_X)
        # transform test dataset
        testX = input_scaler.transform(test_X)
        return trainX, validX, testX
    else:
        return train_X, val_X, test_X

def evaluate_model(nn_design, train_X, train_y, val_X, val_y, test_X, test_y):
	# define model
    model = Sequential()
    
    layers = len(nn_design)
    
    for i in range(1, layers):
        # for all layers, activation function is ReLU
        if i < layers-1:
            act = 'relu'
        # for final layer, activation function is softmax
        else:
            act = 'softmax'
            
        # adding the layers of NN
        model.add(Dense(output_dim = nn_design[i], 
                    input_dim = nn_design[i-1],
                    activation = act,
                    kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)))
    
	# compile model
    opt = optimizers.adam(learning_rate = LEARNING_RATE)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
	
    # fit model
    history = model.fit(train_X, train_y, epochs = EPOCHS, batch_size = BATCH_SIZE, 
          validation_data = (val_X, val_y),
          # callbacks = [EarlyStopping(monitor='val_accuracy', patience=20)]
          ) 

	# evaluate the model
    _, test_acc = model.evaluate(test_X, test_y)
    
    train_cost = history.history['loss']
    train_acc = history.history['accuracy']
    val_cost = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    return (train_cost, train_acc, val_cost, val_acc, test_acc)

(L1, L2) = (0, 1e-1)
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 256

NN_DESIGNS = [(INPUT_NODES, OUTPUT_NODES), # no hidden layer
              (INPUT_NODES, 50, OUTPUT_NODES), # 1 hidden layer
              (INPUT_NODES, 500, OUTPUT_NODES),
              (INPUT_NODES, 1000, OUTPUT_NODES),
              (INPUT_NODES, 50, 50, OUTPUT_NODES), # 2 hidden layers
              (INPUT_NODES, 50, 1000, OUTPUT_NODES),
              (INPUT_NODES, 500, 50, OUTPUT_NODES),
              (INPUT_NODES, 500, 500, OUTPUT_NODES),
              (INPUT_NODES, 500, 1000, OUTPUT_NODES),
              (INPUT_NODES, 1000, 50, OUTPUT_NODES),
              (INPUT_NODES, 1000, 500, OUTPUT_NODES),
              (INPUT_NODES, 1000, 1000, OUTPUT_NODES)]

# start off by standardizing/normalizing our data
(trainX, valX, testX) = transform_dataset(StandardScaler(), nn_x_train, nn_x_validation, test_x)

# preparting plots
fig1, ax1 = plt.subplots(2,6, figsize = (40,20))
fig2, ax2 = plt.subplots(2,6, figsize = (40,20))

fig1.suptitle("Change in cost", fontsize = 40)
fig2.suptitle("Change in accuracy", fontsize = 40)

# train each neural network architecture and plot the change in cost and accuracy
for i, design in enumerate(NN_DESIGNS):
    (train_cost, train_acc, val_cost, val_acc, test_acc) = evaluate_model(design, trainX, nn_y_train, valX, nn_y_validation, testX, test_y_01)
    
    ep = list(range(1,EPOCHS+1))
     
    ax1[i//6, i%6].plot(ep, train_cost, 'ro-', label = 'Train')
    ax1[i//6, i%6].plot(ep, val_cost, 'ko-', label = 'Val')
    ax1[i//6, i%6].set_xlabel('EPOCH', fontsize = 20)
    ax1[i//6, i%6].set_ylabel('cost', fontsize = 20)
    ax1[i//6, i%6].set_title(str(design), fontsize = 25)
    ax1[i//6, i%6].legend(fontsize = 20)
    
    ax2[i//6, i%6].plot(ep, train_acc, 'ro-', label = 'Train')
    ax2[i//6, i%6].plot(ep, val_acc, 'ko-', label = 'Val')
    ax2[i//6, i%6].plot(ep, [test_acc]*len(ep), 'b-')
    ax2[i//6, i%6].set_xlabel('EPOCH', fontsize = 20)
    ax2[i//6, i%6].set_ylabel('accuracy', fontsize = 20)
    ax2[i//6, i%6].set_ylim(0.5, 1)
    ax2[i//6, i%6].set_title(str(design), fontsize = 25)
    ax2[i//6, i%6].legend(fontsize = 20)
plt.show()