# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:03:59 2020

@author: Zeng Fung

Code to compare the effects of scaled vs unscaled input data on the image 
classification accuracy in Neural Networks
"""

PERCENTAGE_OF_TRAIN = 0.8

import pandas as pd
import numpy as np
import data_augmentation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers #for l1 or l2 regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.constraints import maxnorm
import matplotlib.pyplot as plt

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

# increase number of training data via data augmentation
(train_x, train_y) = data_augmentation.increase_data(train_x, train_y, (128,128))

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

LAYER1_NODES = 500
LAYER2_NODES = 500

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

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

def evaluate_model(train_X, train_y, val_X, val_y, test_X, test_y):
	# define model
    model = Sequential([
        #input to first hidden layer
        Dense(output_dim = LAYER1_NODES, input_dim = INPUT_NODES,
              activation = 'relu', kernel_constraint= maxnorm(4),
              kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)
              ),
                
        #first hidden layer to second hidden layer
        Dense(output_dim = LAYER2_NODES, input_dim = LAYER1_NODES, 
              activation = 'relu', kernel_constraint= maxnorm(4),
              kernel_regularizer = regularizers.l1_l2(l1 = L1, l2 = L2)
              ),
                
        #second hidden layer to output
        Dense(output_dim = OUTPUT_NODES, input_dim = LAYER2_NODES, activation = 'softmax'),
        ])
    
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

def run_model(input_scaler, train_X, train_y, val_X, val_y, test_X, test_y):
	# get dataset
    trainX, valX, testX = transform_dataset(input_scaler, train_X, val_X, test_X)
	# repeated evaluation of model
    result = evaluate_model(trainX, train_y, valX, val_y, testX, test_y)
    return result

L1_list = [0, 1e-2, 1e-4]
L2_list = [0, 1e-2, 1e-4]
  
fig1, ax1 = plt.subplots(3,3, figsize = (40,20))
fig2, ax2 = plt.subplots(3,3, figsize = (40,20))
fig1.suptitle("Change in cost", fontsize = 40)
fig2.suptitle("Change in accuracy", fontsize = 40)
    
for i, L1 in enumerate(L1_list):
    for j, L2 in enumerate(L2_list):
        (none_train_cost, none_train_acc, none_val_cost, none_val_acc, none_test_acc) = run_model(None, nn_x_train, nn_y_train, nn_x_validation, nn_y_validation, test_x, test_y_01)
        (minmax_train_cost, minmax_train_acc, minmax_val_cost, minmax_val_acc, minmax_test_acc) = run_model(MinMaxScaler(), nn_x_train, nn_y_train, nn_x_validation, nn_y_validation, test_x, test_y_01)
        (std_train_cost, std_train_acc, std_val_cost, std_val_acc, std_test_acc) = run_model(StandardScaler(), nn_x_train, nn_y_train, nn_x_validation, nn_y_validation, test_x, test_y_01)
        
        ep = list(range(1,EPOCHS+1))
     
        # plot cost functions
        ax1[i,j].plot(ep, none_train_cost, 'r:', label = 'None (Train)')
        ax1[i,j].plot(ep, none_val_cost, 'ro-', label = 'None(Val)')
        ax1[i,j].plot(ep, minmax_train_cost, 'b:', label = 'MinMax (Train)')
        ax1[i,j].plot(ep, minmax_val_cost, 'bo-', label = 'MinMax (Val)')
        ax1[i,j].plot(ep, std_train_cost, 'k:', label = 'Std (Train)')
        ax1[i,j].plot(ep, std_val_cost, 'ko-', label = 'Std (Val)')
        ax1[i,j].set_xlabel('EPOCH', fontsize = 20)
        ax1[i,j].set_ylabel('cost', fontsize = 20)
        ax1[i,j].set_title('L1='+str(L1)+',L2='+str(L2), fontsize = 25)
        ax1[i,j].legend(fontsize = 20)
        
        # plot accuracy functions
        ax2[i,j].plot(ep, none_train_acc, 'r:', label = 'None (Train)')
        ax2[i,j].plot(ep, none_val_acc, 'ro-', label = 'None(Val)')
        ax2[i,j].plot(ep, [none_test_acc]*len(ep), 'r-')
        ax2[i,j].plot(ep, minmax_train_acc, 'b:', label = 'MinMax (Train)')
        ax2[i,j].plot(ep, minmax_val_acc, 'bo-', label = 'MinMax (Val)')
        ax2[i,j].plot(ep, [minmax_test_acc]*len(ep), 'b-')
        ax2[i,j].plot(ep, std_train_acc, 'k:', label = 'Std (Train)')
        ax2[i,j].plot(ep, std_val_acc, 'ko-', label = 'Std (Val)')
        ax2[i,j].plot(ep, [std_test_acc]*len(ep), 'k-')
        ax2[i,j].set_xlabel('EPOCH', fontsize = 20)
        ax2[i,j].set_ylabel('accuracy', fontsize = 20)
        ax2[i,j].set_ylim(0.4, 1)
        ax2[i,j].set_title('L1='+str(L1)+',L2='+str(L2), fontsize = 25)
        ax2[i,j].legend(fontsize = 20)
plt.show()