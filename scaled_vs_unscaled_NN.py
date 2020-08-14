# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:03:59 2020

@author: LZFun
"""

PERCENTAGE_OF_TRAIN = 0.8

import pandas as pd
import random
import math
import numpy as np
import time

print('Reading file...')
# import mask.csv
full_data = pd.read_csv('mask.csv', header = 0)

# split into two separate data frames by images with and without masks
with_mask = full_data[full_data.with_mask == 'Yes']
without_mask = full_data[full_data.with_mask == 'No']

# randomly select training data index
# the amount of training data to be selected depends on the value input into the variable PERCENTAGE_OF_TRAIN
trainidx_withmask = random.sample(range(len(with_mask)), math.floor(PERCENTAGE_OF_TRAIN * len(with_mask)))
trainidx_withoutmask = random.sample(range(len(without_mask)), math.floor(PERCENTAGE_OF_TRAIN * len(without_mask)))

# remaining data will be used as testing data
testidx_withmask = list(set(list(range(len(with_mask)))) - set(trainidx_withmask))
testidx_withoutmask = list(set(list(range(len(without_mask)))) - set(trainidx_withoutmask))

# build a data frame of training data
train_withmask = with_mask.iloc[trainidx_withmask, :]
train_withoutmask = without_mask.iloc[trainidx_withoutmask, :]
train_data = pd.concat([train_withmask, train_withoutmask], axis = 0).reset_index(drop = True)

# build a data frame of testing data
test_withmask = with_mask.iloc[testidx_withmask, :]
test_withoutmask = without_mask.iloc[testidx_withoutmask, :]
test_data = pd.concat([test_withmask, test_withoutmask], axis = 0).reset_index(drop = True)

# converting the data frame into arrays to be used to run ML algorithms
train_x = np.array(train_data.iloc[:, 1:])
train_y = np.array(train_data.iloc[:, 0])

test_x = np.array(test_data.iloc[:, 1:])
test_y_true = np.array(test_data.iloc[:, 0])

classification_result = {}

# delete variables that are no longer needed
del with_mask
del without_mask
del trainidx_withmask
del trainidx_withoutmask
del testidx_withmask
del testidx_withoutmask
del train_withmask
del train_withoutmask
del test_withmask
del test_withoutmask

#%%
print("Setting up variables for NN")
(NUM_OF_DATA, INPUT_NODES) = train_x.shape
OUTPUT_NODES = 2

from sklearn.model_selection import train_test_split

# convert test and training y values from 'Yes' or 'No' to [1,0] or [0,1] respectively
train_y_01 = np.zeros((NUM_OF_DATA,2), dtype = int)
for i in range(NUM_OF_DATA):
    if train_y[i] == 'Yes':
        train_y_01[i][0] = 1
    else:
        train_y_01[i][1] = 1
test_y_01 = np.zeros((len(test_y_true), 2), dtype = int)
for i in range(len(test_y_true)):
    if test_y_true[i] == 'Yes':
        test_y_01[i][0] = 1
    else:
        test_y_01[i][1] = 1
        
# splitting training data into train and validation data sets
(nn_x_train, nn_x_validation, nn_y_train, nn_y_validation) = train_test_split(train_x, train_y_01, train_size = 0.8, test_size = 0.2, random_state = 1)

#%%
print("Running NN")

LAYER1_NODES = 500
LAYER2_NODES = 500

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras import regularizers #for l1 or l2 regularizers
from keras.callbacks import EarlyStopping #stop training when monitored argument stops decreasing/increasing
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

def evaluate_model(train_X, train_y, val_X, val_y, test_X, test_y):
    start = time.time()
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
    
    end = time.time()
    time_taken = end - start
    
    train_cost = history.history['loss']
    train_acc = history.history['accuracy']
    val_cost = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    return (train_cost, train_acc, val_cost, val_acc, test_acc, time_taken)

def run_model(input_scaler, train_X, train_y, val_X, val_y, test_X, test_y):
	# get dataset
    trainX, valX, testX = transform_dataset(input_scaler, train_X, val_X, test_X)
	# repeated evaluation of model
    result = evaluate_model(trainX, train_y, valX, val_y, testX, test_y)
    return result

L1_list = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
L2_list = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
  
results = {}

for j, L1 in enumerate(L1_list):
    fig1, ax1 = plt.subplots(2,3, figsize = (40,20))
    fig2, ax2 = plt.subplots(2,3, figsize = (40,20))
    for i, L2 in enumerate(L2_list):
        print('Running L1 =',L1, 'L2 =',L2)
        (none_train_cost, none_train_acc, none_val_cost, none_val_acc, none_test_acc, none_time) = run_model(None, nn_x_train, nn_y_train, nn_x_validation, nn_y_validation, test_x, test_y_01)
        (minmax_train_cost, minmax_train_acc, minmax_val_cost, minmax_val_acc, minmax_test_acc, minmax_time) = run_model(MinMaxScaler(), nn_x_train, nn_y_train, nn_x_validation, nn_y_validation, test_x, test_y_01)
        (std_train_cost, std_train_acc, std_val_cost, std_val_acc, std_test_acc, std_time) = run_model(StandardScaler(), nn_x_train, nn_y_train, nn_x_validation, nn_y_validation, test_x, test_y_01)
        
        res_none = {'transformation' : 'None',
                    'L1' : L1,
                    'L2' : L2,
                    'test_acc' : none_test_acc}
        res_minmax = {'transformation' : 'MinMax',
                    'L1' : L1,
                    'L2' : L2,
                    'test_acc' : minmax_test_acc}
        res_std = {'transformation' : 'Std',
                    'L1' : L1,
                    'L2' : L2,
                    'test_acc' : std_test_acc}
        res_all = {'None' : res_none,
                   'MinMax' : res_minmax,
                   'Std' : res_std}
        
        results[(L1, L2)] = res_all 
        
        ep = list(range(1,EPOCHS+1))
     
        ax1[i//3, i%3].plot(ep, none_train_cost, 'r:', label = 'None (Train)')
        ax1[i//3, i%3].plot(ep, none_val_cost, 'ro-', label = 'None(Val)')
        ax1[i//3, i%3].plot(ep, minmax_train_cost, 'b:', label = 'MinMax (Train)')
        ax1[i//3, i%3].plot(ep, minmax_val_cost, 'bo-', label = 'MinMax (Val)')
        ax1[i//3, i%3].plot(ep, std_train_cost, 'k:', label = 'Std (Train)')
        ax1[i//3, i%3].plot(ep, std_val_cost, 'ko-', label = 'Std (Val)')
        ax1[i//3, i%3].set_xlabel('EPOCH', fontsize = 20)
        ax1[i//3, i%3].set_ylabel('cost', fontsize = 20)
        ax1[i//3, i%3].set_title('Change in cost (L1='+str(L1)+',L2='+str(L2)+')', fontsize = 25)
        ax1[i//3, i%3].legend(fontsize = 20)
        
        ax2[i//3, i%3].plot(ep, none_train_acc, 'r:', label = 'None (Train)')
        ax2[i//3, i%3].plot(ep, none_val_acc, 'ro-', label = 'None(Val)')
        ax2[i//3, i%3].plot(ep, [none_test_acc]*len(ep), 'r-')
        ax2[i//3, i%3].plot(ep, minmax_train_acc, 'b:', label = 'MinMax (Train)')
        ax2[i//3, i%3].plot(ep, minmax_val_acc, 'bo-', label = 'MinMax (Val)')
        ax2[i//3, i%3].plot(ep, [minmax_test_acc]*len(ep), 'b-')
        ax2[i//3, i%3].plot(ep, std_train_acc, 'k:', label = 'Std (Train)')
        ax2[i//3, i%3].plot(ep, std_val_acc, 'ko-', label = 'Std (Val)')
        ax2[i//3, i%3].plot(ep, [std_test_acc]*len(ep), 'k-')
        ax2[i//3, i%3].set_xlabel('EPOCH', fontsize = 20)
        ax2[i//3, i%3].set_ylabel('accuracy', fontsize = 20)
        ax2[i//3, i%3].set_ylim(0.4, 1)
        ax2[i//3, i%3].set_title('Change in accuracy (L1='+str(L1)+',L2='+str(L2)+')', fontsize = 25)
        ax2[i//3, i%3].legend(fontsize = 20)
    fig1.show()
    fig2.show()

  
