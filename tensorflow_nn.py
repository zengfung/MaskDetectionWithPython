# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:43 2020

@author: LZFun
"""

PERCENTAGE_OF_TRAIN = 0.9

import pandas as pd
import random
import math as m
import numpy as np

print('Reading file...')
# import mask.csv
full_data = pd.read_csv('mask.csv', header = 0)

# split into two separate data frames by images with and without masks
with_mask = full_data[full_data.with_mask == 'Yes']
without_mask = full_data[full_data.with_mask == 'No']

# randomly select training data index
# the amount of training data to be selected depends on the value input into the variable PERCENTAGE_OF_TRAIN
trainidx_withmask = random.sample(range(len(with_mask)), m.floor(PERCENTAGE_OF_TRAIN * len(with_mask)))
trainidx_withoutmask = random.sample(range(len(without_mask)), m.floor(PERCENTAGE_OF_TRAIN * len(without_mask)))

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


#%%
import tensorflow.compat.v1 as tf
import random
import time

print('Running NN')
(NUM_OF_DATA, INPUT_NODES) = train_x.shape
LAYER1_NODES = 2000
LAYER2_NODES = 50
OUTPUT_NODES = 2

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

LEARNING_RATE = 5e-4
EPOCHS = 100
BATCH_SIZE = 100

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
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(layer2_to_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

start = time.time()
with tf.Session() as sess:
   # initialise the variables
    sess.run(init_op)
    total_batch = int(NUM_OF_DATA / BATCH_SIZE)
    
    prev_cost = float('inf')
    
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
            
        if prev_cost <= avg_cost:
            print('Cost Increased')
        prev_cost = avg_cost
        
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print("Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y_01}))
    test_y_pred_01 = sess.run(layer2_to_output, feed_dict = {x : test_x, y : test_y_01})
    print(sess.run(accuracy, feed_dict={x: test_x, y: test_y_01}))
end = time.time()
time_taken = end - start

print('Time taken:', round(time_taken,2), 'seconds')



