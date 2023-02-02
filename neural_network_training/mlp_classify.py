# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
from numpy import loadtxt
from keras import Sequential
from tensorflow.keras.utils import to_categorical

from keras.datasets import mnist

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
x_train = train_dataset[:,0:metadata].astype(float)
y_train = train_dataset[:,metadata:cols].astype(int)

# hide (remove) user metadata after first eight metadata
x_train = np.delete(x_train, slice(umeta_end, umeta_hide_end), 1)
# hide (remove) resource metadata after first eight resource metadata
x_train = np.delete(x_train, slice(rmeta_end, rmeta_hide_end), 1)
if debug:
  print('User/resource metadata after meta data removal:', x_train.shape[1])

# split x, y from test dataset
x_test = test_dataset[:,0:metadata].astype(float)
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

print(x_train[0].shape)

# Create the model
model = Sequential()
model.add(Dense(32, input_shape=(16,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
#model.summary()

outputFileName = 'mlp'
DIR_ASSETS = 'results/'
PATH_MODEL = DIR_ASSETS + outputFileName + '.hdf5'

# Configure the model and start training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
history = model.fit(x_train, y_train, epochs=60, batch_size=250, verbose=1, validation_split=0.2)

if debug:
  print('Saving trained mlp to {}.'.format(PATH_MODEL))
if not os.path.isdir(DIR_ASSETS):
    os.mkdir(DIR_ASSETS)
model.save(PATH_MODEL)

#save history to separate file
import pickle

PATH_HISTORY_FILE = DIR_ASSETS + 'history_' + outputFileName
with open(PATH_HISTORY_FILE, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

RESULT_FILE = DIR_ASSETS + 'result.txt'
result_file = open(RESULT_FILE, 'w+')
result_file.write('train data file name:%s\n' % (trainDataFileName))
result_file.write('test data file name:%s\n' % (testDataFileName))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
'ResNet%d' % (depth)
result_file.write('Test loss:%f\n' % (scores[0]))
print('Test accuracy:', scores[1])
result_file.write('Test accuracy:%f\n' % (scores[1]))

# measure True Positive/ Negative, False Positive/ Negative
from sklearn import metrics
from sklearn.metrics import precision_score, confusion_matrix

y_preds = model.predict(x_test)
y_preds = (y_preds > 0.5).astype(int)

g_tn = 0
g_fp = 0
g_fn = 0
g_tp = 0

# Measure True Positive/ Negative, False Positive/ Negative for each operation, 
# then combine it to measure actual counts
# we calculate the FPR, FNR offline
print('True Positive/ Negative, False Positive/ Negative Information')
for i in range(4):
  tn, fp, fn, tp = confusion_matrix(y_test[:, i:i+1], y_preds[:, i:i+1]).ravel()
  print('op%d  # tn: %s, fp: %s, fn: %s, tp: %s' % (i+1, tn, fp, fn, tp))
  g_tn = g_tn + tn
  g_fp = g_fp + fp
  g_fn = g_fn + fn
  g_tp = g_tp + tp
print('All operations # tn: %s, fp: %s, fn: %s, tp: %s' % (g_tn, g_fp, g_fn, g_tp))
result_file.write('TN: %s, FP: %s, FN: %s, TP: %s' % (g_tn, g_fp, g_fn, g_tp))
result_file.close()