import sys, os

import numpy as np
from numpy import expand_dims

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D


## This function was given
def pad_image(image, max_size=(25, 25)):
    """
    Simply pad an image with zeros up to max_size.
    """
    size = np.shape(image)
    px, py = (max_size[0]-size[0]), (max_size[1]-size[1])
    a1 = int(np.floor(px/2.0))
    a2 = int(np.ceil(px/2.0))
    a3 = int(np.floor(py/2.0))
    a4 = int(np.ceil(py/2.0))
    image = np.pad(image, ((a1, a2), (a3, a4)), 'constant', constant_values=(0))
    #image = np.pad(image, (map(int,((np.floor(px/2.), np.ceil(px/2.)))), map(int,(np.floor(py/2.), np.ceil(py/2.)))), 'constant')
    return image


## This function was given
def normalize(histo, multi=255):
    """
    Normalize picture in [0,multi] range, with integer steps. E.g. multi=255 for 256 steps.
    """
    return (histo/np.max(histo)*multi).astype(int)


# Loading input data
data0 = np.load('qcd_leading_jet.npz', allow_pickle=True, encoding='bytes')
data1 = np.load('top_leading_jet.npz', allow_pickle=True, encoding='bytes')

qcd_data = data0['arr_0']
top_data = data1['arr_0']

# I want to use 40K events from each sample (total-x=40K)
data0 = np.delete(qcd_data, np.s_[1:15001], 0)
data1 = np.delete(top_data, np.s_[1:18722], 0)

print('We have {} QCD jets and {} top jets'.format(len(data0), len(data1)))

# combine data and labels
x_data = np.concatenate((data0, data1))
y_data = np.array([0]*len(data0)+[1]*len(data1))

x_data = list(x_data)

# pad and normalize images
x_data = list(map(pad_image, x_data))
x_data = list(map(normalize, x_data))

# shuffle data
np.random.seed(0)  # for reproducibility
x_data, y_data = np.random.permutation(np.array([x_data, y_data]).T).T


# the data coming out of previous commands is a list of 2D arrays. We want a 3D np array (n_events, xpixels, ypixels)
x_data = np.stack(x_data)
x_data = expand_dims(x_data, axis=3)
y_data = keras.utils.to_categorical(y_data, 2)

# Data Loading Technique 1: Split training and testing data from both QCD and Top jets combined.
# Using 50,000 events for training and the remaining for testing. This is a general-purpose split of the dataset.
def load_data1():
    n_train = 50000
    (x_train, x_test) = x_data[:n_train], x_data[n_train:]
    (y_train, y_test) = y_data[:n_train], y_data[n_train:]
    return x_train, x_test, y_train, y_test


# Data Loading Technique 2: QCD jets used entirely for training, and Top jets used entirely for testing.
# This method is ideal for scenarios where the model should learn from one type and generalize to another.
def load_data2():
    # Use QCD data for training
    x_train = np.stack(list(map(normalize, map(pad_image, qcd_data))))
    x_train = expand_dims(x_train, axis=3)
    y_train = np.zeros(len(x_train))  # Label 0 for QCD jets

    # Use top jet data for testing
    x_test = np.stack(list(map(normalize, map(pad_image, top_data))))
    x_test = expand_dims(x_test, axis=3)
    y_test = np.ones(len(x_test))  # Label 1 for top jets

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    return x_train, x_train, y_train, y_train


# Data Loading Technique 3: QCD jets used entirely for training. Testing is done with a mixed set of QCD and Top jets.
# Useful for evaluating model performance on balanced classes, simulating a mix of known and unknown data types.
def load_data3():
    # Take equal amounts of QCD and top data for testing
    min_data_length = min(len(qcd_data), len(top_data))

    # Training Data - QCD Only
    x_train = np.stack(list(map(normalize, map(pad_image, qcd_data[:min_data_length]))))
    x_train = expand_dims(x_train, axis=3)
    y_train = np.zeros(len(x_train))  # Label 0 for QCD jets

    # Testing Data - Equal mix of QCD and Top
    mixed_test_data = np.concatenate((qcd_data[:min_data_length], top_data[:min_data_length]))
    x_test = np.stack(list(map(normalize, map(pad_image, mixed_test_data))))
    x_test = expand_dims(x_test, axis=3)

    # Creating labels for mixed test data
    y_test = np.array([0] * min_data_length + [1] * min_data_length)

    # Shuffling the mixed test data
    np.random.seed(0)  # For reproducibility
    indices = np.arange(len(mixed_test_data))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    return x_train, x_test, y_train, y_test


# Data Loading Technique 4: QCD jets used entirely for both training and testing, essentially overfitting to QCD.
# This is useful for baseline checks to ensure model can fit data when training and testing on the same set.
def load_data4():
    # Use QCD data for training
    x_train = np.stack(list(map(normalize, map(pad_image, qcd_data))))
    x_train = expand_dims(x_train, axis=3)
    y_train = np.zeros(len(x_train))  # Label 0 for QCD jets

    # Use top jet data for testing
    x_test = np.stack(list(map(normalize, map(pad_image, qcd_data))))
    x_test = expand_dims(x_test, axis=3)
    y_test = np.ones(len(x_test))  # Label 1 for top jets

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    return x_train, x_train, y_train, y_train
