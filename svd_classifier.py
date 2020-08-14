PERCENTAGE_OF_TRAIN = 0.8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy as sc

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


def classify_svd_training(train_mat, train_class):
    X = train_mat.T
    y = train_class.T
    
    U = [[], []]
    S = [[], []]
    V = [[], []]
    for i, class_val in enumerate(['Yes', 'No']):
        index = (y == class_val)
        matrix = X[:, index]
        (U[i], S[i], V[i]) = sc.linalg.svd(matrix, full_matrices = False)
    return (U, S, V)        

def classify_test(test_mat, n, U):
    X = test_mat.T
    test_size = X.shape[1]
    classification = []
    for i in range(test_size):
        b = X[:, i]
        resnorm = np.empty(2)
        for j in range(2):
            A = U[j][:,:n]
            x = np.linalg.inv(A.T @ A) @ A.T @ b
            res = b - A @ x
            resnorm[j] = np.linalg.norm(res, 2) / np.linalg.norm(b, 2)
        if resnorm[0] < resnorm[1]:
            classification.append('Yes')
        else:
            classification.append('No')
    return classification

# splitting train data into train + validation data
(svd_train_x, svd_val_x, svd_train_y, svd_val_y) = train_test_split(train_x, train_y, train_size = 0.8, test_size = 0.2, random_state = 1)

# obtaining SVD of training data
(train_U, train_S, train_V) = classify_svd_training(svd_train_x, svd_train_y)

# plot of magnitude of singular values
fig, ax = plt.subplots(1,3, figsize = (15, 5))

x1 = list(range(1,len(train_S[0])+1))
y1 = train_S[0]
ax[0].plot(x1, y1, label = 'Yes')

x2 = list(range(1, len(train_S[1])+1))
y2 = train_S[1]
ax[0].plot(x2, y2, label = 'No')

ax[0].set_xlabel('k-th Singular Value')
ax[0].set_ylabel('s')
ax[0].title.set_text('Original Values (s)')
ax[0].legend()

x1 = list(range(1,len(train_S[0])+1))
y1 = np.log(train_S[0])
ax[1].plot(x1, y1, label = 'Yes')

x2 = list(range(1, len(train_S[1])+1))
y2 = np.log(train_S[1])
ax[1].plot(x2, y2, label = 'No')

ax[1].set_xlabel('k-th Singular Value')
ax[1].set_ylabel('log(s)')
ax[1].title.set_text('Log Transformed Values (log(s))')
ax[1].legend()

x1 = list(range(1,151))
y1 = np.log(train_S[0][:150])
ax[2].plot(x1, y1, label = 'Yes')

# plot of accuracy of validation data vs k-singular values
time_taken = []
accuracy = []
n = list(range(5,601,5))

for i in n:
    start = time.time()
    val_y_pred = classify_test(svd_val_x, i, train_U)
    end = time.time()
    time_taken.append(round(end - start, 2))
    confmat = confusion_matrix(svd_val_y, val_y_pred, labels = ['Yes', 'No'])
    acc = (confmat[0][0] + confmat[1][1]) / np.sum(confmat)
    accuracy.append(acc)

fig, ax1 = plt.subplots(figsize = (15,5))

color = 'tab:red'
ax1.set_xlabel('k')
ax1.set_ylabel('Time taken', color=color)
ax1.plot(n, time_taken, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(n, accuracy, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.7,1])

fig.suptitle('Time taken vs Accuracyof Validation Data based on k', y = 1)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
x2 = list(range(1, 151))
y2 = np.log(train_S[1][:150])
ax[2].plot(x2, y2, label = 'No')

ax[2].set_xlabel('k-th Singular Value')
ax[2].set_ylabel('log(s)')
ax[2].title.set_text('Log Transformed First 150 Values (log(s))')
ax[2].legend()

fig.show()

# classifying test data
k_singularval = (accuracy.index(max(accuracy)) + 1) * 5
test_y_pred = classify_test(test_x, k_singularval, train_U)

confmat = confusion_matrix(test_y_true, test_y_pred, labels = ['Yes', 'No'])
print(confmat)