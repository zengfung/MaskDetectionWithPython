# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:44:44 2020

@author: Zeng Fung

View transformed images based on the pixel values in the data frame from 
full_mask.csv file
"""

import pandas as pd
from PIL import Image as pimg
import numpy as np

df = pd.read_csv('full_mask.csv', header = 0)

#%%
IMAGE_TO_DISPLAY = 5
IMAGE_SIZE = (200,200)

img_vec = df.iloc[IMAGE_TO_DISPLAY,1:]
img_vec = img_vec.to_numpy(dtype = int)
img_mat = img_vec.reshape(IMAGE_SIZE, order = 'F')
img = pimg.fromarray(img_mat)
img.show()


