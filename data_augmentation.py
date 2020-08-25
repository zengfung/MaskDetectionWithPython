# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:11:04 2020

@author: Zeng Fung

Increase the number of data/images by augmenting existing images

The methods of image augmentation used:
    - mirror images
    - upside-down images
    - upside-down mirror images
    - images rotated to the right
    - mirror images rotated to the right
    - images rotated to the left
    - mirror images rotated to the left
"""

import numpy as np
import sys

def increase_data(X, y, size):
    # check if length of X and y are equal
    if not len(X) == len(y):
        sys.exit("Length of X and y are different!")
    else:
        (count, pixels) = np.shape(X)
    
    # define new sets of X and y
    X_new = np.zeros((8*count, pixels), dtype = int)
    y_new = np.empty((8*count,), dtype = object)
    for i in range(count):
        # set range of index for augmentation of image i
        start = 8 * i
        end = 8 * (i + 1)
        # augmenting image i
        (data, cls) = (X[i,:].copy(), y[i])
        (X_new[start:end, :], y_new[start:end]) = augment(data, cls, size)
    return (X_new, y_new)
    
        
def augment(img_vec, img_class, img_size):
    img_mat = img_vec.reshape(img_size)
    # mirrored image
    img_mirror = np.fliplr(img_mat)
    img_mirror_data = np.asarray(img_mirror)
    img_mirror_data = img_mirror_data.reshape(-1)
    # # flipped image
    img_flip = np.flipud(img_mat)
    img_flip_data = np.asarray(img_flip)
    img_flip_data = img_flip_data.reshape(-1)
    # flipped + mirrored image
    img_flip_mirror = np.flipud(img_mirror)
    img_flip_mirror_data = np.asarray(img_flip_mirror)
    img_flip_mirror_data = img_flip_mirror_data.reshape(-1)
    # left-rotate image
    img_lrot = np.rot90(img_mat, 3)
    img_lrot_data = np.asarray(img_lrot)
    img_lrot_data = img_lrot_data.reshape(-1)
    # left-rotate + flip image
    img_lrot_flip = np.flipud(img_lrot)
    img_lrot_flip_data = np.asarray(img_lrot_flip)
    img_lrot_flip_data = img_lrot_flip_data.reshape(-1)
    # right-rotate image
    img_rrot = np.rot90(img_mat)
    img_rrot_data = np.asarray(img_rrot)
    img_rrot_data = img_rrot_data.reshape(-1)
    # right-rotate + flip image
    img_rrot_flip = np.flipud(img_rrot)
    img_rrot_flip_data = np.asarray(img_rrot_flip)
    img_rrot_flip_data = img_rrot_flip_data.reshape(-1)
    # stacking vectors of all augmented images into a single array
    output_X = np.vstack((img_vec, img_mirror_data,
                          img_flip_data, img_flip_mirror_data,
                          img_lrot_data, img_lrot_flip_data,
                          img_rrot_data, img_rrot_flip_data))
    output_y = np.full((8,), img_class, dtype = object)
    return (output_X, output_y)