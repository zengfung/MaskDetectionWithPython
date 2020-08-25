# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:44:44 2020

@author: Zeng Fung

View transformed images based on the pixel values in the data frame from 
full_mask.csv file
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mask.csv', header = 0)

IMAGE_TO_DISPLAY = 7
IMAGE_SIZE = (128,128,3)
MODE = 'P'

img_vec = df.iloc[IMAGE_TO_DISPLAY,1:]
img_vec = img_vec.to_numpy(dtype = int)
img_mat = img_vec.reshape(IMAGE_SIZE)
plt.imshow(img_mat)
plt.show()