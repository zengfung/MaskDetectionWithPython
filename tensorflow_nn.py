# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:43 2020

@author: Zeng Fung

Code to run Feed-Forward Neural Network in Tensorflow
"""

PERCENTAGE_OF_TRAIN = 0.8

import pandas as pd
import numpy as np
import data_augmentation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

import tensorflow.compat.v1 as tf
import random

print('Running NN')
(NUM_OF_DATA, INPUT_NODES) = train_x.shape
LAYER1_NODES = 2000
LAYER2_NODES = 50
OUTPUT_NODES = 2

# splitting training data into train and validation data sets
(train_x, val_x, train_y, val_y) = train_test_split(train_x, train_y, train_size = 0.8, test_size = 0.2, random_state = 1)

# increase number of training data
(train_x, train_y) = data_augmentation.increase_data(train_x, train_y, (128,128,3))

# convert test and training y values from 'Yes' or 'No' to [1,0] or [0,1] respectively
onehot_encoder = OneHotEncoder(sparse=False)
train_y_01 = train_y.reshape(len(train_y),1)
train_y_01 = onehot_encoder.fit_transform(train_y_01)
val_y_01 = val_y.reshape(len(val_y),1)
val_y_01 = onehot_encoder.fit_transform(val_y_01)
test_y_01 = test_y_true.reshape(len(test_y_true),1)
test_y_01 = onehot_encoder.fit_transform(test_y_01)

LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 1024

# input layer
x = tf.placeholder(tf.float32, [None, INPUT_NODES])
# output layer
y = tf.placeholder(tf.float32, [None, OUTPUT_NODES])

# weights and bias for arcs from input to layer 1
W1 = tf.Variable(tf.random_normal([INPUT_NODES, LAYER1_NODES], stddev = 1e-5), name = 'W1')
b1 = tf.Variable(tf.random_normal([LAYER1_NODES]), name = 'b1')
# weights and bias for arcs from layer 1 to layer 2
W2 = tf.Variable(tf.random_normal([LAYER1_NODES, LAYER2_NODES], stddev = 1e-5), name = 'W2')
b2 = tf.Variable(tf.random_normal([LAYER2_NODES]), name = 'b2')
# weights and bias for arcs from layer 2 to output
W3 = tf.Variable(tf.random_normal([LAYER2_NODES, OUTPUT_NODES], stddev = 1e-5), name = 'W3')
b3 = tf.Variable(tf.random_normal([OUTPUT_NODES]), name = 'b3')

# going from one layer to the other via F(x) = ReLu(W*x + b)
input_to_layer1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
layer1_to_layer2 = tf.nn.relu(tf.add(tf.matmul(input_to_layer1, W2), b2))
layer2_to_output = tf.nn.softmax(tf.add(tf.matmul(layer1_to_layer2, W3), b3))

# cost function
y_clipped = tf.clip_by_value(layer2_to_output, 1e-8, 0.999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
optimizer = opt.minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
prediction = tf.argmax(layer2_to_output, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(layer2_to_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# converting one-hot predictions to 'Yes'/'No' classes
def onehot_to_cat(predY):
    result = []
    for i in range(len(predY)):
        if predY[i] == 1:
            result.append('Yes')
        else:
            result.append('No')
    return result
 

with tf.Session() as sess:
   # initialise the variables
    sess.run(init_op)
    total_batch = int(NUM_OF_DATA / BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        avg_cost = 0
        randomized_index = random.sample(list(range(NUM_OF_DATA)), NUM_OF_DATA)
        for i in range(total_batch):
            batch_x = train_x[randomized_index[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)], :]
            batch_y = train_y_01[randomized_index[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)], :]
            _, c = sess.run([optimizer, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            output = sess.run(layer2_to_output, feed_dict = {x : batch_x, y : batch_y})
            avg_cost += c / total_batch
            
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print("Accuracy:", sess.run(accuracy, feed_dict={x: val_x, y: val_y_01}))
    test_y_pred_01 = sess.run(prediction, feed_dict = {x : test_x, y : test_y_01})
    test_y_pred = onehot_to_cat(test_y_pred_01)
    confmat = confusion_matrix(test_y_true, test_y_pred, labels = ['Yes', 'No'])
    print(confmat)
    print("Test accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y_01}))


