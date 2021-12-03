# Written By Nalin Ahuja (nalinahuja), Rohan Narasimha (novablazerx)

import os
import sys

# Set TensorFlow Logging Level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# End Imports------------------------------------------------------------------------------------------------------------------------------------------------------------

# Print Format Strings
NL, TB, CR = "\n", "\t", "\r"

# Embedded Resource Paths
FACE_DETECTION_MODEL = "./detection"
FACE_MAPPING_MODEL = "./mapping"
IMAGE_MASK_PATH = "./masks"

# Video Capture Resolution (px, px)
VIDEO_CAPTURE_RESOLUTION = (1280, 720)

# Model Mapping Resolution (px, px)
MODEL_MAPPING_RESOLUTION = (96, 96)

# Selected Mask Name
SELECTED_MASK = "glasses.png"

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
    """
    # Load Face Detection Model
    face_detector = keras.models.load_model(os.path.join(FACE_DETECTION_MODEL, "saved"))
    """

    # TESTING ONLY
    face_detector = cv.CascadeClassifier(os.path.join(FACE_DETECTION_MODEL, "saved", "detection_model.xml"))
except Exception as e:
    # Raise FileNotFoundError
    raise FileNotFoundError("could not load facial detection model")

def detect_faces(image):
    # Convert Image Colorspace To Grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # TESTING ONLY
    faces = face_detector.detectMultiScale(image, 1.1, 4)

    # Return Face Coordinates
    return (faces)

# End Facial Detection Model Loading-------------------------------------------------------------------------------------------------------------------------------------

# Print Status
print("Loading facial mapping model..." + NL)

try:
    # Load Facial Mapping Model
    face_mapper = keras.models.load_model(os.path.join(FACE_MAPPING_MODEL, "saved"))
except Exception as e:
    # Raise FileNotFoundError
    raise FileNotFoundError("could not load facial mapping model")

def map_face(image, x, y, w, h):
    # Convert Image Colorspace To Grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Resize Frame To Mappping Dimensions
    image = cv.resize(image, dsize = MODEL_MAPPING_RESOLUTION, interpolation = cv.INTER_AREA)

    # Normalize Image Pixel Values
    image = np.divide(image, 255)

    # Expand Pixel Value Dimension
    image = np.expand_dims(image, axis = -1)

    # Expand Image Frame Dimension
    image = np.expand_dims(image, axis = 0)

    # Run Image Through Facial Mapping Model
    keypoints = face_mapper.predict(image)[0]

    # Scale Keypoints To Fit Image Dimensions
    keypoints = np.multiply(keypoints, w)

    # Round Keypoint Coordinates
    keypoints = np.round(keypoints)

    # Cast Keypoint Values
    keypoints = keypoints.astype(np.int32)

    # Initialize Feature Dictionary
    features = dict()

    # Set Nose Tip Coordinates
    features["nose_tip"] = (keypoints[0] + x, keypoints[1] + y)

    # Set Left Eye Coordinates
    features["left_eye"] = (keypoints[2] + x, keypoints[3] + y)

    # Set Right Eye Coordinates
    features["right_eye"] = (keypoints[4] + x, keypoints[5] + y)

    # Set Mouth Center Coodinates
    features["mouth_center"] = (keypoints[6] + x, keypoints[7] + y)

    # Return Featues Dictionary
    return (features)

# End Facial Mapping Model Loading---------------------------------------------------------------------------------------------------------------------------------------

# Print Status
print("Loading \"%s\" image mask..." % str(SELECTED_MASK) + NL)

# Form Mask Path
mask_path = os.path.join(IMAGE_MASK_PATH, SELECTED_MASK)

# Verify Mask Path
if (not(os.path.exists(mask_path))):
    # Raise FileNotFoundError
    raise FileNotFoundError("mask file does not exist")

# Initialize Global Image Mask
mask = cv.imread(mask_path, -1)

# Conditionally Initialize Overlay Function
if (SELECTED_MASK == "glasses.png"):
    def overlay_mask(fr, features):
        # Set Mask Scope To Global
        global mask

        # TODO
        pass
elif (SELECTED_MASK == "nose.png"):
    def overlay_mask(fr, features):
        # Set Mask Scope To Global
        global mask

        # TODO
        pass

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
        # Verify Frame Dimensions
        if ((fr.shape[0] > 0) and (fr.shape[1] > 0)):
            # Mirror Frame About Y Axis
            fr = cv.flip(fr, 1)

            # Resize Video Frame To Fixed Dimensions
            fr = cv.resize(fr, dsize = VIDEO_CAPTURE_RESOLUTION, interpolation = cv.INTER_AREA)

            # Run Video Frame Through Facial Detector
            faces = detect_faces(fr)

            # Iterate Over Faces In Video Frame
            for x, y, w, h in (faces):
                # Crop Into Facial Region
                face = fr[y : y + h, x : x + w]

                # Verify Face Dimensions
                if (not(face.shape[0] > 0) or not(face.shape[1] > 0)):
                    # Skip Face
                    continue

                # Run Facial Region Through Facial Mapper
                features = map_face(face, x, y, w, h)

                # Overlay Image Masks Using Feature Coordinates
                fr = overlay_mask(fr, features)

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
