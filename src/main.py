# Written By Nalin Ahuja (nalinahuja), Rohan Narasimha (novablazerx)

import os
import sys

# Set TensorFlow Logging Level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# End Imports------------------------------------------------------------------------------------------------------------------------------------------------------------

# Print Format Strings
NL, CR = "\n", "\r"

# TensorFlow Model Paths
FACE_DETECTION_MODEL = "./detection"
FACE_MAPPING_MODEL = "./mapping"

# End Embedded Constants-------------------------------------------------------------------------------------------------------------------------------------------------

import random
import cv2 as cv
import numpy as np
import tensorflow as tf

from sklearn import metrics
from tensorflow import keras

# TensorFlow Settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# End Module Imports-----------------------------------------------------------------------------------------------------------------------------------------------------

try:
    # Load Face Detection Model
    FACE_DETECTOR = keras.models.load_model(os.path.join(FACE_DETECTION_MODEL, "saved"))
except Exception as e:
    # Raise FileNotFoundError
    raise FileNotFoundError("could not load facial detection model")

try:
    # Load Facial Mapping Model
    FACE_MAPPER = keras.models.load_model(os.path.join(FACE_MAPPING_MODEL, "saved"))
except Exception as e:
    # Raise FileNotFoundError
    raise FileNotFoundError("could not load facial mapping model")

# Load Models------------------------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == "__main__"):
    # Create Video Stream
    stream = cv.VideoCapture(0)

    # Read Frame From Stream
    ret, fr = stream.read()

    # Process Frames
    while (ret):
        # Convert Grame Colorspace To Grayscale
        gs = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)

        # TODO: Run Frame Through Facial Detector
            # TODO: If bounding boxes exist, map each face in a bounding box

            # TODO: Convert frame to black and white
            
            # TODO: Add features to face on bounding box

        # Show Video Frame
        cv.imshow("video", fr)

        # Check For ESCAPE Key
        if (cv.waitKey(1) == 27):
            # Break Loop
            break

        # Read Frame From Stream
        ret, fr = stream.read()

    # Release Frame Stream
    stream.release()

# End Main---------------------------------------------------------------------------------------------------------------------------------------------------------------
