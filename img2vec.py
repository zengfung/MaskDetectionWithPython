# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:11:04 2020

@author: Zeng Fung

Transform all images into vectors and store them into a data frame and export
them into .csv files
"""

import os
from PIL import Image as pimg
from PIL import ImageOps
import numpy as np
import pandas as pd

# Variables to play around/edit if necessary
CONVERTED_IMAGE_SIZE = (128, 128)
CONVERTED_IMAGE_MODE = 'P' #choose either '1','L', 'P', 'RGB', 'RGBA'
CONVERTED_IMAGE_TYPE = '.jpg'

with_mask_folder = './with_mask/'
without_mask_folder = './without_mask/'

with_mask_files = os.listdir(with_mask_folder)
without_mask_files = os.listdir(without_mask_folder)

# Preliminary analysis on with_mask images
with_mask_dict = {'mode' : {}, 'type' : {}, 'image_size' : {}}
(img_x, img_y) = (0,0)
for image_file in with_mask_files:
    [filename, filetype] = image_file.split('.')
    if not filetype in with_mask_dict['type']:
        with_mask_dict['type'][filetype] = 0
    with_mask_dict['type'][filetype] += 1
    
    cur_image = pimg.open(with_mask_folder + image_file)
    if not cur_image.mode in with_mask_dict['mode']:
        with_mask_dict['mode'][cur_image.mode] = 0
    with_mask_dict['mode'][cur_image.mode] += 1
    
    if not cur_image.size in with_mask_dict['image_size']:
        with_mask_dict['image_size'][cur_image.size] = 0
    with_mask_dict['image_size'][cur_image.size] += 1
    img_x += cur_image.size[0]
    img_y += cur_image.size[1]
with_mask_dict['image_size']['average'] = (img_x / len(with_mask_files), img_y / len(with_mask_files))

# with_mask images preliminary results
print("--with_mask images--")
print('Number of files:', len(with_mask_files))
print('Number of modes:', len(with_mask_dict['mode']))
print(with_mask_dict['mode'])
print('Number of types:', len(with_mask_dict['type']))
print(with_mask_dict['type'])
print('Number of different sizes:', len(with_mask_dict['image_size']))
print('Average image size:', with_mask_dict['image_size']['average'])
print('____________________________________')
   
# Preliminary analysis on without_mask images
without_mask_dict = {'mode' : {}, 'type' : {}, 'image_size' : {}}
(img_x, img_y) = (0,0)
for image_file in without_mask_files:
    [filename, filetype] = image_file.split('.')
    if not filetype in without_mask_dict['type']:
        without_mask_dict['type'][filetype] = 0
    without_mask_dict['type'][filetype] += 1
    
    cur_image = pimg.open(without_mask_folder + image_file)
    if not cur_image.mode in without_mask_dict['mode']:
        without_mask_dict['mode'][cur_image.mode] = 0
    without_mask_dict['mode'][cur_image.mode] += 1
    
    if not cur_image.size in without_mask_dict['image_size']:
        without_mask_dict['image_size'][cur_image.size] = 0
    without_mask_dict['image_size'][cur_image.size] += 1
    img_x += cur_image.size[0]
    img_y += cur_image.size[1]
without_mask_dict['image_size']['average'] = (img_x / len(with_mask_files), img_y / len(with_mask_files))

print('--without_mask images--')
print('Number of files:', len(without_mask_files))
print('Number of modes:', len(without_mask_dict['mode']))
print(without_mask_dict['mode'])
print('Number of types:', len(without_mask_dict['type']))
print(without_mask_dict['type'])
print('Number of different sizes:', len(without_mask_dict['image_size']))
print('Average image size:', without_mask_dict['image_size']['average'])
print('____________________________________________')

# obtaining with_mask image data in the form of matrix
num_of_img = len(with_mask_files)
if CONVERTED_IMAGE_MODE in ['1', 'L', 'P']:
    img_size = CONVERTED_IMAGE_SIZE[0] * CONVERTED_IMAGE_SIZE[1]
elif CONVERTED_IMAGE_MODE == 'RGB':
    img_size = CONVERTED_IMAGE_SIZE[0] * CONVERTED_IMAGE_SIZE[1] * 3
elif CONVERTED_IMAGE_MODE == 'RGBA':
    img_size = CONVERTED_IMAGE_SIZE[0] * CONVERTED_IMAGE_SIZE[1] * 4
else:
    img_size = input('What is the size of the image?')
    
with_mask_data = np.empty((num_of_img*8, img_size), dtype = int)
index = 0
for image_file in with_mask_files:
    cur_image = pimg.open(with_mask_folder + image_file)
    if cur_image.mode != CONVERTED_IMAGE_MODE:
        cur_image = cur_image.convert(CONVERTED_IMAGE_MODE)
    # normal image
    convert_img = cur_image.resize(CONVERTED_IMAGE_SIZE)
    img_data = np.asarray(convert_img)
    img_data = img_data.reshape(-1)
    # mirrored image
    img_mirror = ImageOps.mirror(convert_img)
    img_mirror_data = np.asarray(img_mirror)
    img_mirror_data = img_mirror_data.reshape(-1)
    # flipped image
    img_flip = ImageOps.flip(convert_img)
    img_flip_data = np.asarray(img_flip)
    img_flip_data = img_flip_data.reshape(-1)
    # flipped + mirrored image
    img_flip_mirror = ImageOps.flip(img_mirror)
    img_flip_mirror_data = np.asarray(img_flip_mirror)
    img_flip_mirror_data = img_flip_mirror_data.reshape(-1)
    # left-rotate image
    img_lrot = convert_img.rotate(angle = 90)
    img_lrot_data = np.asarray(img_lrot)
    img_lrot_data = img_lrot_data.reshape(-1)
    # left-rotate + flip image
    img_lrot_flip = ImageOps.flip(img_lrot)
    img_lrot_flip_data = np.asarray(img_lrot_flip)
    img_lrot_flip_data = img_lrot_flip_data.reshape(-1)
    # right-rotate image
    img_rrot = convert_img.rotate(angle = -90)
    img_rrot_data = np.asarray(img_rrot)
    img_rrot_data = img_rrot_data.reshape(-1)
    # right-rotate + flip image
    img_rrot_flip = ImageOps.flip(img_rrot)
    img_rrot_flip_data = np.asarray(img_rrot_flip)
    img_rrot_flip_data = img_rrot_flip_data.reshape(-1)
    # inserting into data frame
    with_mask_data[index] = img_data
    with_mask_data[index + 1] = img_mirror_data
    with_mask_data[index + 2] = img_flip_data
    with_mask_data[index + 3] = img_flip_mirror_data
    with_mask_data[index + 4] = img_lrot_data
    with_mask_data[index + 5] = img_lrot_flip_data
    with_mask_data[index + 6] = img_rrot_data
    with_mask_data[index + 7] = img_rrot_flip_data
    index += 8
    
# obtaining without_mask image_data in the form of matrix
num_of_img = len(without_mask_files)
if CONVERTED_IMAGE_MODE in ['1', 'L', 'P']:
    img_size = CONVERTED_IMAGE_SIZE[0] * CONVERTED_IMAGE_SIZE[1]
elif CONVERTED_IMAGE_MODE == 'RGB':
    img_size = CONVERTED_IMAGE_SIZE[0] * CONVERTED_IMAGE_SIZE[1] * 3
elif CONVERTED_IMAGE_MODE == 'RGBA':
    img_size = CONVERTED_IMAGE_SIZE[0] * CONVERTED_IMAGE_SIZE[1] * 4
else:
    img_size = input('What is the size of the image?')
    
without_mask_data = np.empty((num_of_img*8, img_size), dtype = int)
index = 0
for image_file in without_mask_files:
    cur_image = pimg.open(without_mask_folder + image_file)
    if cur_image.mode != CONVERTED_IMAGE_MODE:
        cur_image = cur_image.convert(CONVERTED_IMAGE_MODE)
    # normal image
    convert_img = cur_image.resize(CONVERTED_IMAGE_SIZE)
    img_data = np.asarray(convert_img)
    img_data = img_data.reshape(-1)
    # mirrored image
    img_mirror = ImageOps.mirror(convert_img)
    img_mirror_data = np.asarray(img_mirror)
    img_mirror_data = img_mirror_data.reshape(-1)
    # flipped image
    img_flip = ImageOps.flip(convert_img)
    img_flip_data = np.asarray(img_flip)
    img_flip_data = img_flip_data.reshape(-1)
    # flipped + mirrored image
    img_flip_mirror = ImageOps.flip(img_mirror)
    img_flip_mirror_data = np.asarray(img_flip_mirror)
    img_flip_mirror_data = img_flip_mirror_data.reshape(-1)
    # left-rotate image
    img_lrot = convert_img.rotate(angle = 90)
    img_lrot_data = np.asarray(img_lrot)
    img_lrot_data = img_lrot_data.reshape(-1)
    # left-rotate + flip image
    img_lrot_flip = ImageOps.flip(img_lrot)
    img_lrot_flip_data = np.asarray(img_lrot_flip)
    img_lrot_flip_data = img_lrot_flip_data.reshape(-1)
    # right-rotate image
    img_rrot = convert_img.rotate(angle = -90)
    img_rrot_data = np.asarray(img_rrot)
    img_rrot_data = img_rrot_data.reshape(-1)
    # right-rotate + flip image
    img_rrot_flip = ImageOps.flip(img_rrot)
    img_rrot_flip_data = np.asarray(img_rrot_flip)
    img_rrot_flip_data = img_rrot_flip_data.reshape(-1)
    # inserting into data frame
    without_mask_data[index] = img_data
    without_mask_data[index + 1] = img_mirror_data
    without_mask_data[index + 2] = img_flip_data
    without_mask_data[index + 3] = img_flip_mirror_data
    without_mask_data[index + 4] = img_lrot_data
    without_mask_data[index + 5] = img_lrot_flip_data
    without_mask_data[index + 6] = img_rrot_data
    without_mask_data[index + 7] = img_rrot_flip_data
    index += 8

# converting everything into a dataframe
col_names = []
for i in range(img_size):
    var_name = 'X' + str(i + 1)
    col_names.append(var_name)
    
with_mask_df = pd.DataFrame(with_mask_data, columns = col_names)
with_mask_df.insert(0, "with_mask", len(with_mask_df) * ['Yes'], False)

without_mask_df = pd.DataFrame(without_mask_data, columns = col_names)
without_mask_df.insert(0, "with_mask", len(without_mask_df) * ['No'], False)

full_mask_df = pd.concat([with_mask_df, without_mask_df], axis = 0)

# exporting data frame into csv file
print('Exporting full_mask_df...')
full_mask_df.to_csv('mask.csv', index = False, header = True)