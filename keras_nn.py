# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:35:00 2020

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

#%%
print("Setting up variables for NN")
(NUM_OF_DATA, INPUT_NODES) = train_x.shape
OUTPUT_NODES = 2
LAYER1_NODES = 8000
LAYER2_NODES = 1000

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
randomized_index = random.sample(list(range(NUM_OF_DATA)), NUM_OF_DATA)
training_num = math.ceil(0.8 * len(randomized_index))

nn_x_train = train_x[randomized_index[:training_num], :]
nn_y_train = train_y_01[randomized_index[:training_num], :]
nn_x_validation = train_x[randomized_index[training_num:], :]
nn_y_validation = train_y_01[randomized_index[training_num:], :]

TRAINING_COUNT = training_num
VALIDATION_COUNT = NUM_OF_DATA - TRAINING_COUNT

print("Running NN")

from keras import initializers, optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers #for l1 or l2 regularizers
from keras.layers.core import Dropout #for dropout to improve efficiency and speed
from keras.callbacks import LearningRateScheduler, EarlyStopping #stop training when monitored argument stops decreasing/increasing

EPOCHS = 30
BATCH_SIZE = 100
LAMBDA = 1e-5
LEARNING_RATE = 1e-5
weight_initializer = initializers.TruncatedNormal(mean = 0, stddev = 1e-5)

start = time.time()
model = Sequential([
    #input to first hidden layer
    Dense(output_dim = LAYER1_NODES, input_dim = INPUT_NODES,
          # kernel_initializer = weight_initializer, bias_initializer = 'zeros',
          activation = 'relu',
          # kernel_regularizer = regularizers.l2(LAMBDA)
          ),
    Dropout(0.25),
    
    #first hidden layer to second hidden layer
    Dense(output_dim = LAYER2_NODES, input_dim = LAYER1_NODES, 
          # kernel_initializer = weight_initializer, bias_initializer = 'zeros',          
          activation = 'relu',
          # kernel_regularizer = regularizers.l2(LAMBDA)
          ),
    Dropout(0.25),
    
    #second hidden layer to output
    Dense(output_dim = OUTPUT_NODES, input_dim = LAYER2_NODES, activation = 'softmax'),
    ])

# compile models with the learning rates set
opt = optimizers.adam(learning_rate = LEARNING_RATE)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

model.fit(nn_x_train, nn_y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, 
          validation_data = (nn_x_validation, nn_y_validation),
          # callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
          ) 

#evaluate model on test set
_, accuracy = model.evaluate(test_x, test_y_01)
end = time.time()
time_taken = end - start

print('Test Data Accuracy:', '{:.3f}'.format(accuracy))
print('Time taken:', round(time_taken,2), 'seconds')

print('\a')

#%%
print("Showing results")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_y_pred_01 = model.predict(test_x, verbose = 1)

test_y_pred = []
for i in range(len(test_y_pred_01)):
    if test_y_pred_01[i][0] > test_y_pred_01[i][1]:
        test_y_pred.append('Yes')
    else:
        test_y_pred.append('No')
 
# plot confusion matrix and record relevant data
fig, ax = plt.subplots(1,2, figsize = (10,5))

for i, normal in enumerate([None, 'true']):
    confmat = confusion_matrix(test_y_true, test_y_pred, labels = ['Yes', 'No'], normalize = normal)
    disp = ConfusionMatrixDisplay(confmat, display_labels = ['Yes', 'No'])
    disp.plot(ax = ax[i], cmap = plt.cm.Blues)
    if i == 0:
        disp.ax_.set_title('Non-normalized Confusion Matrix')
        classification_result['Neural Network'] = {'Time' : time_taken,
                                     'True Positive' : confmat[0][0],
                                     'True Negative' : confmat[1][1],
                                     'False Positive' : confmat[1][0],
                                     'False Negative' : confmat[0][1]}
    else:
        disp.ax_.set_title('Normalized Confusion Matrix')
        disp.im_.set_clim(0,1)