# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
from numpy import loadtxt

from sklearn import metrics
import os
import sys

debug = True

param_count = len(sys.argv)

trainDataFileName = str(sys.argv[1])
testDataFileName = str(sys.argv[2])

batch_size = 16  # trained all networks with batch_size=16

# format of the dataset
# <uid rid> <8-13 user-metadata values> <8-13 resource-metadata values> <4 operations>
# load the train dataset
raw_train_dataset = loadtxt(trainDataFileName, delimiter=' ', dtype=str)
cols = raw_train_dataset.shape[1]
train_dataset = raw_train_dataset[:,2:cols] # TO SKIP UID RID

# load the test dataset
raw_test_dataset = loadtxt(testDataFileName, delimiter=' ', dtype=str)
test_dataset = raw_test_dataset[:,2:cols] # TO SKIP UID RID

# columns after removing uid/rid
cols = train_dataset.shape[1]
if debug:
  print('Total columns:', cols)

# determine number of metadata to be hide
# we will expose first eight user and first eight resource metadata to the model
# there are four operations
# 8 + 8 + 4 = 20

if cols > 20:
    hide_meta_data = cols - 20
else:
    hide_meta_data = 0
print('metadata to be hide: ', hide_meta_data)

# Compute depth and number of epochs based on metadata hide
# We use more deeper network for the dataset where metadata needs to hide
# If the dataset needs to hide metadata, then the depth of network is 56, otherwise 8
# The value of n helps to determine the depth of network
if hide_meta_data > 0:
    n = 9
else:
    n = 1

depth = n * 6 + 2

# we need less epoch for the deeper network
if depth > 8:
  epochs = 30
else:
  epochs = 60

# Model name, depth and version
model_type = 'ResNet%d' % (depth)
if debug:
  print('ResNet model type:', model_type)


# number of metadata
metadata = cols - 4

umeta_end = 8
rmeta_end = 16
umeta_hide_end = umeta_end + hide_meta_data
rmeta_hide_end = rmeta_end + hide_meta_data

# split x, y from train dataset
x_train = train_dataset[:,0:metadata]
y_train = train_dataset[:,metadata:cols].astype(int)

# hide (remove) user metadata after first eight metadata
x_train = np.delete(x_train, slice(umeta_end, umeta_hide_end), 1)
# hide (remove) resource metadata after first eight resource metadata
x_train = np.delete(x_train, slice(rmeta_end, rmeta_hide_end), 1)
if debug:
  print('User/resource metadata after meta data removal:', x_train.shape[1])

# split x, y from test dataset
x_test = test_dataset[:,0:metadata]
y_test = test_dataset[:,metadata:cols].astype(int)

# hide (remove) user/resource metadata after first eight of user/resource metadata
x_test = np.delete(x_test, slice(umeta_end, umeta_hide_end), 1)
x_test = np.delete(x_test, slice(rmeta_end, rmeta_hide_end), 1)
if debug:
  print('User/resource metadata after meta data removal:', x_test.shape[1])

############### OneHot ENCODING ##############
#x_train = to_categorical(x_train)
#x_test = to_categorical(x_test)

if debug:
  print('shape of x_train after encoding', x_train.shape)
  print('shape of x_test after encoding', x_test.shape)
#######################################

#determine batch size
batch_size = min(x_train.shape[0]/10, batch_size)
if debug:
  print('batch size: ' + str(batch_size))

# adding an extra dimension to make the input appropriate for ResNet
#x_train = x_train[..., np.newaxis]
#x_test = x_test[..., np.newaxis]

if debug:
  print('shape of x_train after adding new dimension', x_train.shape)
  print('shape of x_test after adding new dimension', x_test.shape)

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel
y_train_new = [int("".join(str(x) for x in y), 2) for y in y_train]
y_test_new = [int("".join(str(x) for x in y), 2) for y in y_test]
#Train the model using the training sets
clf.fit(x_train, y_train_new)


#Predict the response for test dataset
y_pred_new = clf.predict(x_test)

y_preds = [ [int(x) for x in '{0:04b}'.format(y)] for  y in y_pred_new ]

g_tn = 0
g_fp = 0
g_fn = 0
g_tp = 0

# Measure True Positive/ Negative, False Positive/ Negative for each operation, 
# then combine it to measure actual counts
# we calculate the FPR, FNR offline

from sklearn.metrics import confusion_matrix

RESULT_PATH = './results'
#os.makedirs(RESULT_PATH)

result_file=open(RESULT_PATH + '/results.txt', 'w+')
y_preds = np.array(y_preds)
print('True Positive/ Negative, False Positive/ Negative Information')
for i in range(4):
  tn, fp, fn, tp = confusion_matrix(y_test[:, i:i+1], y_preds[:, i:i+1]).ravel()
  print('op%d  # tn: %s, fp: %s, fn: %s, tp: %s' % (i+1, tn, fp, fn, tp))
  g_tn = g_tn + tn
  g_fp = g_fp + fp
  g_fn = g_fn + fn
  g_tp = g_tp + tp

accuracy = (g_tp + g_tn)/(g_tp + g_tn + g_fp + g_fn)

print('Test accuracy:', accuracy)
result_file.write('Test accuracy:%f\n' % (accuracy))
print('All operations # tn: %s, fp: %s, fn: %s, tp: %s' % (g_tn, g_fp, g_fn, g_tp))
result_file.write('TN: %s, FP: %s, FN: %s, TP: %s\n' % (g_tn, g_fp, g_fn, g_tp))


# train accuracy stuff
y_pred_train = clf.predict(x_train)
y_preds = [ [int(x) for x in '{0:04b}'.format(y)] for  y in y_pred_train ]
y_preds = np.array(y_preds)

g_tn = 0
g_fp = 0
g_fn = 0
g_tp = 0

for i in range(4):
  tn, fp, fn, tp = confusion_matrix(y_train[:, i:i+1], y_preds[:, i:i+1]).ravel()
  print('op%d  # tn: %s, fp: %s, fn: %s, tp: %s' % (i+1, tn, fp, fn, tp))
  g_tn = g_tn + tn
  g_fp = g_fp + fp
  g_fn = g_fn + fn
  g_tp = g_tp + tp

accuracy = (g_tp + g_tn)/(g_tp + g_tn + g_fp + g_fn)

print('train accuracy:', accuracy)
result_file.write('Train accuracy:%f\n' % (accuracy))


result_file.close()