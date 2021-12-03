# Written By Nalin Ahuja (nalinahuja), Rohan Narasimha (novablazerx)

import os
import sys

# Set TensorFlow Logging Level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# End Imports------------------------------------------------------------------------------------------------------------------------------------------------------------

# Print Format Strings
NL, TB, CR = "\n", "\t", "\r"

# TensorFlow Model Paths
FACE_DETECTION_MODEL = "./detection"
FACE_MAPPING_MODEL = "./mapping"

# Video Capture Resolution
VIDEO_CAPTURE_RESOLUTION = (1280, 720)

# Image Mask Names
MASKS = ["glasses.png", "nose.png", "tongue.png"]

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

# Print Status
print("Loading facial detection model..." + NL)

try:
    # Load Face Detection Model
    face_detector = keras.models.load_model(os.path.join(FACE_DETECTION_MODEL, "saved"))
except Exception as e:
    # Raise FileNotFoundError
    raise FileNotFoundError("could not load facial detection model")

def detect_face(image):
    # TODO: Implement This Wrapper

# End Facial Detection Model Loading-------------------------------------------------------------------------------------------------------------------------------------

# Print Status
print("Loading facial mapping model..." + NL)

try:
    # Load Facial Mapping Model
    face_mapper = keras.models.load_model(os.path.join(FACE_MAPPING_MODEL, "saved"))
except Exception as e:
    # Raise FileNotFoundError
    raise FileNotFoundError("could not load facial mapping model")

def map_face(image):
    # Convert Image Colorspace To Grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Normalize Image Pixel Values
    image = np.divide(image, 255)

    # Expand Pixel Value Dimension
    image = np.expand_dims(image, axis = -1)

    # Expand Image Frame Dimension
    image = np.expand_dims(image, axis = 0)

    # Run Image Through Facial Mapping Model
    keypoints = face_mapper.predict(image)[0]

    # Scale Keypoints To Fit Image Dimensions
    keypoints = np.multiply(keypoints, image.shape[0])
    
    # Round Keypoint Coordinates  
    keypoints = np.round(keypoints)

    # Cast Keypoint Values
    keypoints = keypoints.astype(np.int32)

    # Initialize Feature Dictionary
    features = dict()

    # Set Nose Tip Coordinates
    features["nose_tip"] = tuple(keypoints[0 : 2])

    # Set Left Eye Coordinates
    features["left_eye"] = tuple(keypoints[2 : 4])

    # Set Right Eye Coordinates
    features["right_eye"] = tuple(keypoints[4 : 6])

    # Set Mouth Center Coodinates
    features["mouth_center"] = tuple(keypoints[6 : 8])

    # Return Featues Dictionary
    return (features)

# End Facial Mapping Model Loading---------------------------------------------------------------------------------------------------------------------------------------

def create_mask():
    pass

def 

# End Image Masking Functions--------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == "__main__"):
    # Print Status
    print("Starting video capture..." + NL)

    # Create Video Stream
    stream = cv.VideoCapture(0)

    # Read Frame From Stream
    ret, fr = stream.read()

    # Process Frames
    while (ret):
        # Resize Video Frame To Fixed Dimensions
        fr = cv.resize(fr, dsize = VIDEO_CAPTURE_RESOLUTION, interpolation = cv.INTER_AREA)

        # TODO: Run Video Frame Through Facial Detector

        # TODO: Verify Faces Exist In Frame

            # TODO: Map Each Face In Frame Using Bounding Box

            # TODO: Add Masks To Face On Original Image Using Facial Map

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
