# Written By Nalin Ahuja (nalinahuja), Rohan Narasimha (novablazerx)

import os
import sys

# Set TensorFlow Logging Level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# End Module Imports----------------------------------------------------------------------------------------------------------------------------------------------------

# Random Seed Value
SEED_VALUE = 1618

# Print Format Strings
NL, TB, CR = "\n", "\t", "\r"

# TensorFlow Model Paths
TF_DATA_PATH = "./data"
TF_SAVE_PATH = "./saved"

# Training Data Size
TRAIN_SIZE = 0.80

# TensorFlow Model Hyperparameters
TF_DROP_OUT = 0.25
TF_NUM_EPOCHS = 10
TF_BATCH_SIZE = 32
TF_LEARNING_RATE = 0.01

# Set Input Dimensions
INPUT_X = 96
INPUT_Y = 96
INPUT_Z = 1

# Evaluation Sample Size
EVALUATION_SIZE = 5

# End Embedded Constants------------------------------------------------------------------------------------------------------------------------------------------------

import random
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import metrics
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Set Deterministic Random Seeds
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# TensorFlow Settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# End Module Imports----------------------------------------------------------------------------------------------------------------------------------------------------

def build_tf_conv_net(x_train, y_train, x_test, y_test, eps = TF_NUM_EPOCHS, lr = TF_LEARNING_RATE, bs = TF_BATCH_SIZE, drop_out = True, drop_rate = TF_DROP_OUT):
    # Initialize New Sequential Model Instance
    model = keras.Sequential()

    # Add Convolutional Network Layers
    model.add(keras.layers.Conv2D(32, kernel_size = [3, 3], activation = tf.nn.tanh, input_shape = [INPUT_X, INPUT_Y, INPUT_Z]))
    model.add(keras.layers.Conv2D(32, kernel_size = [3, 3], activation = tf.nn.tanh))

    # Add Pooling And Normalization Layers
    model.add(keras.layers.MaxPooling2D(pool_size = [2, 2]))
    model.add(keras.layers.BatchNormalization())

    # Add Convolutional Network Layers
    model.add(keras.layers.Conv2D(64, kernel_size = [3, 3], activation = tf.nn.tanh))
    model.add(keras.layers.Conv2D(64, kernel_size = [3, 3], activation = tf.nn.tanh))

    # Add Pooling And Normalization Layers
    model.add(keras.layers.MaxPooling2D(pool_size = [2, 2]))
    model.add(keras.layers.BatchNormalization())

    # Add Flattening Layer
    model.add(keras.layers.Flatten())

    # Add Dense Layer
    model.add(keras.layers.Dense(256, activation = tf.nn.tanh))

    # Check Dropout Setting
    if (drop_out):
        # Add Dropout Layer
        model.add(keras.layers.Dropout(drop_rate, input_shape = [2]))

    # Add Dense Layer
    model.add(keras.layers.Dense(128, activation = tf.nn.tanh))

    # Check Dropout Setting
    if (drop_out):
        # Add Dropout Layer
        model.add(keras.layers.Dropout(drop_rate, input_shape = [2]))

    # Add Output Layer
    model.add(keras.layers.Dense(y_train.shape[1], activation = tf.nn.sigmoid))

    # Initialize Loss Function
    loss_func = keras.losses.mse

    # Initialize Model Optimizer
    opt_func = tf.optimizers.Adam(learning_rate = lr)

    # Compile Model
    model.compile(loss = loss_func, optimizer = opt_func, metrics = ["accuracy"])

    # Train Model
    model.fit(x_train, y_train, epochs = eps, batch_size = bs, validation_data = (x_test, y_test), verbose = True)

    # Print Separator
    print(NL * 1, end = "")

    # Return Model
    return (model)

# End Classifier Functions----------------------------------------------------------------------------------------------------------------------------------------------

def get_data():
    # Check For Data Path
    if (not(os.path.exists(TF_DATA_PATH))):
        # Raise Error
        raise FileNotFoundError("data path missing")

    # Display Status
    print(CR + "Loading dataset...", end = "")

    # TODO: Write Image Loading Here

    # Split Facial Detection Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, train_size = TRAIN_SIZE)

    # Convert Input Data Into Numpy Arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Convert Output Data Into Numpy Arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Display Information About Dataset
    print(CR + "Shape of x_train dataset: %s" % str(x_train.shape))
    print("Shape of y_train dataset: %s" % str(y_train.shape))
    print("Shape of x_test dataset: %s" % str(x_test.shape))
    print("Shape of y_test dataset: %s" % str(y_test.shape))
    print(NL * 1, end = "")

    # Return Data
    return ((x_train, y_train), (x_test, y_test))

def train_model(train_data, test_data):
    # Unpack Training Data
    x_train, y_train = train_data

    # Unpack Testing Data
    x_test, y_test = test_data

    # Display Status
    print("Training Tensorflow convolutional network..." + NL)

    # Return Model
    return (build_tf_conv_net(x_train, y_train, x_test, y_test, eps = TF_NUM_EPOCHS))

def run_model(data, model):
    # Display Status
    print("Running Tensorflow convolutional network...")

    # Run TensorFlow Convolutional Model On Data
    preds = model.predict(data)

    # Return Predictions
    return (np.array(preds))

def eval_results(data, y_pred):
    # Unpack Testing Data
    _, y_test = data

    # Randomly Sample Predicted Outputs
    samples = random.sample(range(len(y_pred)), EVALUATION_SIZE)

    # Iterate Over Selected Predicted Outputs
    for i in (samples):
        # Compute Sum Of Squared Output Differences
        l2_error = np.sum(np.square(np.subtract(y_test[i], y_pred[i])))

        # Round Test Output Vector
        y_test[i] = np.round(y_test[i], decimals = 3)

        # Round Predicted Output Vector
        y_pred[i] = np.round(y_pred[i], decimals = 3)

        # Print Sample Index
        print(NL + "Sample Index %d:" % int(i))

        # Print Expected Output
        print(TB + "Test: " + str(y_test[i]))

        # Print Predicted Output
        print(TB + "Pred: " + str(y_pred[i]))

        # Print L2 Error
        print(TB + "L2 Error: %.2f" % float(l2_error))

# End Pipeline Functions------------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == "__main__"):
    # Ignore Numpy Warnings
    np.seterr(all = "ignore")

    # Get Processed Data
    data = get_data()

    # Train Model On Processed Data
    model = train_model(data[0], data[1])

    # Run Model On Processed Data
    preds = run_model(data[1][0], model)

    # Evaluate Model Results
    eval_results(data[1], preds)

    # Check For Model File
    if (os.path.exists(TF_SAVE_PATH)):
        # Get Model Overwrite Confirmation
        if (len(os.listdir(TF_SAVE_PATH)) and input(NL + "Model file exists, confirm overwrite [y/n]: ").lower() != "y"):
            # Print Separator
            print(NL * 1, end = "")
        else:
            # Print Status
            print(NL + "Saving TensorFlow model to disk..." + NL)

            # Save Model To Disk
            model.save(TF_SAVE_PATH)

# End Main Function-----------------------------------------------------------------------------------------------------------------------------------------------------
