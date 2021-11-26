# Written By Nalin Ahuja (nalinahuja), Rohan Narasimha (novablazerx)

import os
import sys

# Set TensorFlow Logging Level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# End Module Imports----------------------------------------------------------------------------------------------------------------------------------------------------

# Random Seed Value
SEED_VALUE = 1618

# Print Format Strings
NL, CR = "\n", "\r"

# TensorFlow Model Paths
TF_DATA_PATH = "./data"
TF_SAVE_PATH = "./saved"

# Training Data Size
TRAIN_SIZE = 0.75

# TensorFlow Model Hyperparameters
TF_DROP_OUT = 0.20
TF_NUM_EPOCHS = 10
TF_LEARNING_RATE = 0.001

# Set Number Of Classes
OUTPUT_SIZE = 2

# Set Input Dimensions
INPUT_Y = 60
INPUT_X = 60
INPUT_Z = 3

# Set Input Size
INPUT_SIZE = INPUT_X * INPUT_Y * INPUT_Z

# End Embedded Constants------------------------------------------------------------------------------------------------------------------------------------------------

import random
import cv2 as cv
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import codecs
from tqdm import tqdm
import requests
from PIL import Image
import io
from io import BytesIO

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

def build_tf_conv_net(x_train, y_train, eps = TF_NUM_EPOCHS, lr = TF_LEARNING_RATE, drop_out = True, drop_rate = TF_DROP_OUT):
    # Initialize New Sequential Model Instance
    model = keras.Sequential()

    # Add Convolutional Network Layers
    model.add(keras.layers.Conv2D(32, kernel_size = [3, 3], activation = tf.nn.relu, input_shape = [INPUT_X, INPUT_Y, INPUT_Z]))
    model.add(keras.layers.Conv2D(32, kernel_size = [3, 3], activation = tf.nn.relu))

    # Add Pooling And Normalization Layers
    model.add(keras.layers.MaxPooling2D(pool_size = [2, 2]))
    model.add(keras.layers.BatchNormalization())

    # Add Convolutional Network Layers
    model.add(keras.layers.Conv2D(64, kernel_size = [3, 3], activation = tf.nn.relu))
    model.add(keras.layers.Conv2D(64, kernel_size = [3, 3], activation = tf.nn.relu))

    # Add Pooling And Normalization Layers
    model.add(keras.layers.MaxPooling2D(pool_size = [2, 2]))
    model.add(keras.layers.BatchNormalization())

    # Add Flattening Layer
    model.add(keras.layers.Flatten())

    # Add Dense Layer
    model.add(keras.layers.Dense(256, activation = tf.nn.relu))

    # Check Dropout Setting
    if (drop_out):
        # Add Dropout Layer
        model.add(keras.layers.Dropout(drop_rate, input_shape = [2]))

    # Add Dense Layer
    model.add(keras.layers.Dense(128, activation = tf.nn.relu))

    # Check Dropout Setting
    if (drop_out):
        # Add Dropout Layer
        model.add(keras.layers.Dropout(drop_rate, input_shape = [2]))

    # Add Output Layer
    model.add(keras.layers.Dense(OUTPUT_SIZE, activation = tf.nn.softmax))

    # Initialize Loss Function
    loss_func = keras.losses.categorical_crossentropy

    # Initialize Model Optimizer
    opt_func = tf.optimizers.Adam(learning_rate = lr)

    # Compile Model
    model.compile(loss = loss_func, optimizer = opt_func, metrics = ["accuracy"])

    # Train Model
    model.fit(x_train, y_train, epochs = eps)

    # Print Separator
    print("NL" * 1, end = "")

    # Return Model
    return (model)

# End Classifier Functions----------------------------------------------------------------------------------------------------------------------------------------------

def encode_preds(preds):
    # Initialize Encoded Predictions Representation
    enc = np.zeros(preds.shape)

    # Determine Indicies With Maximum Probability
    mpi = np.argmax(preds, axis = 1)

    # Iterate Over Predictions
    for i in range(preds.shape[0]):
        # Set Position Of Maximum Probability
        enc[i][mpi[i]] = 1

    # Return Encoded Predictions Representation
    return (enc)

# End Utility Functions------------------------------------------------------------------------------------------------------------------------------------------------

def get_data():
    # Check For Data Path
    if (not(os.path.exists(TF_DATA_PATH))):
        # Raise Error
        raise FileNotFoundError("data path missing")

    # Display Status
    print(CR + "Loading dataset...", end = "")

    # TODO, load data from TF_DATA_PATH
    
    #x = type: ndarray
    #y = type: ndarray
    #https://www.kaggle.com/dataturks/face-detection-in-images
    #x = pd.read_json("data/face_detection.json")
    #f = open("data/face_detection.json")
    #x = json.load(f)
    fname = "data/face_detection.json"
    jsonData = []
    
    with codecs.open(fname, 'rU') as js:
        for line in js:
            jsonData.append(json.loads(line))
            
    x = []
    y = []
    
    print("json data 0 : ", jsonData[0])

    i = 0
    for data in tqdm(jsonData):
        print("i : ", i)
        i = i + 1
        if i < 10:
            continue
        print("data content : ", data['content'])
        response = requests.get(data['content'])
        print("response : ", response.content)
        byteImgIO = io.BytesIO()
        byteImg = Image.open(response.content)
        byteImg.save(byteImgIO, "PNG")
        byteImgIO.seek(0)
        byteImg = byteImgIO.read()
        img = np.asarray(Image.open(BytesIO(byteImg)))
        #images.append([img, data["annotation"]])
        #images.append(img)
        metadata = data["annotation"]
        for data in metadata:
            height = data['imageHeight']
            width = data['imageWidth']
            points = data['points']
            if 'Face' in data['label']:
                start_x = round(width*points[0]['x'])
                start_y = round(height*points[0]['y'])
                end_x = round(width*points[1]['x'])
                end_y = round(height*points[1]['y'])
                target = (start_x, start_y, end_x, end_y)
                x.append(img)
                y.append(np.asarray(target))

    # Split Columnar Spam Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = TRAIN_SIZE)

    # Convert Input Data Into Numpy Arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Convert Output Data Into Numpy Arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Display Information About Dataset
    print("Dataset: %s" % DATASET)
    print("Shape of x_train dataset: %s." % str(x_train.shape))
    print("Shape of y_train dataset: %s." % str(y_train.shape))
    print("Shape of x_test dataset: %s." % str(x_test.shape))
    print("Shape of y_test dataset: %s." % str(y_test.shape))
    print("NL" * 1, end = "")

    # Return Data
    return ((x_train, y_train), (x_test, y_test))

def process_data(raw):
    # Unpack Data From Raw Input
    ((x_train, y_train), (x_test, y_test)) = raw

    # Process Training Input Images
    for i, img in enumerate(x_train):
        # Convert Image Data Type
        img = img.astype("float64")

        # Resize Training Input Image To Specified Dimensions
        img = cv.resize(img, dsize = (INPUT_X, INPUT_Y))

        # Add Dimension To Numpy Array
        img = np.expand_dims(img, axis = 0)

        # Update Training Data Point
        x_train[i] = img

    # Process Testing Input Images
    for i, img in enumerate(x_test):
        # Convert Image Data Type
        img = img.astype("float64")

        # Resize Testing Input Image To Specified Dimensions
        img = cv.resize(img, dsize = (INPUT_X, INPUT_Y))

        # Add Dimension To Numpy Array
        img = np.expand_dims(img, axis = 0)

        # Update Training Data Point
        x_test[i] = img

    # Process Integer Arrays Into Binary Class Matrices
    y_train = keras.utils.to_categorical(y_train, OUTPUT_SIZE)
    y_test = keras.utils.to_categorical(y_test, OUTPUT_SIZE)

    # Display Information About Dataset
    print("New shape of x_train dataset: %s." % str(x_train.shape))
    print("New shape of x_test dataset: %s." % str(x_test.shape))
    print("New shape of y_train dataset: %s." % str(y_train.shape))
    print("New shape of y_test dataset: %s." % str(y_test.shape))
    print("NL" * 1, end = "")

    # Return Preprocessed Data
    return ((x_train, y_train), (x_test, y_test))

def train_model(data):
    # Unpack Training Data
    x_train, y_train = data

    # Display Status
    print("Training Tensorflow convolutional network..." + NL)

    # Return Model
    return (build_tf_conv_net(x_train, y_train, eps = TF_NUM_EPOCHS))

def run_model(data, model):
    # Display Status
    print("Running Tensorflow convolutional network..." + NL)

    # Run TensorFlow Convolutional Model On Data
    preds = model.predict(data)

    # One Hot Encode Predictions
    preds = encode_preds(preds)

    # Return Predictions
    return (np.array(preds))

def eval_results(data, y_pred):
    # Unpack Testing Data
    _, y_test = data

    # Format Test Data
    y_test = np.argmax(y_test, axis = 1)

    # Format Prediction Data
    y_pred = np.argmax(y_preds, axis = 1)

    # Compute Model Accuracy
    acc = metrics.accuracy_score(y_test, y_pred)

    # Display Model Accuracy
    print(NL + "Classifier Accuracy: {:.3f}".format(acc))

    # Generate Model Classification Report
    cr = metrics.classification_report(y_test, y_pred, target_names = ["Positive", "Negative"])

    # Display Model Classification Report
    print(NL + "Classifier Report:")
    print(cr)

    # Generate Confusion Matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Display Confusion Matrix
    print("Confusion Matrix:")
    print(cm)

# End Pipeline Functions------------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == '__main__'):
    # Ignore Numpy Warnings
    np.seterr(all = "ignore")

    # Get Raw Data
    raw = get_data()

    # Process Raw Data
    data = process_data(raw)

    # Train Model On Raw Data
    model = train_model(data[0])

    # Run Model On Raw Data
    preds = run_model(data[1][0], model)

    # Evaluate Model Results
    eval_results(data[1], preds)

    # Check For Model File
    if (os.path.exists(TF_MODEL_PATH)):
        # Get Model Overwrite Confirmation
        if (input(NL + "Model file exists, confirm overwrite [y/n]: ").lower() != "y"):
            # Exit Program
            sys.exit()

        # Print Status
        print(NL + "Saving TensorFlow model to disk..." + NL)

        # Save Model To Disk
        model.save(TF_MODEL_PATH)

# End Main Function-----------------------------------------------------------------------------------------------------------------------------------------------------