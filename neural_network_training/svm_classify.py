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
print(y_train)
y_train_new = [int("".join(str(x) for x in y), 2) for y in y_train]
y_test_new = [int("".join(str(x) for x in y), 2) for y in y_test]
#Train the model using the training sets
clf.fit(x_train, y_train_new)


#Predict the response for test dataset
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test_new, y_pred))
