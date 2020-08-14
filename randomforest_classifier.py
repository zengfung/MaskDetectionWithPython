PERCENTAGE_OF_TRAIN = 0.8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
classifier.fit(train_x, train_y)

test_y_pred = np.array(classifier.predict(test_x), dtype = object)

confmat = confusion_matrix(test_y_true, test_y_pred, labels = ['Yes', 'No'])
print(confmat)