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

# Localization Padding (px)
LOCALIZATION_PADDING = 50

# Video Mapping Resolution (px, px)
VIDEO_MAPPING_RESOLUTION = (96, 96)

# Video Capture Resolution (px, px)
VIDEO_CAPTURE_RESOLUTION = (1280, 720)

# Image Mask Names
IMAGE_MASKS = os.listdir("./masks")

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

def detect_face(image):
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

def map_face(image):
    # Convert Image Colorspace To Grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Resize Frame To Mappping Dimensions
    image = cv.resize(image, dsize = VIDEO_MAPPING_RESOLUTION, interpolation = cv.INTER_AREA)

    # Normalize Image Pixel Values
    image = np.divide(image, 255)

    # Expand Pixel Value Dimension
    image = np.expand_dims(image, axis = -1)

    # Expand Image Frame Dimension
    image = np.expand_dims(image, axis = 0)

    # Run Image Through Facial Mapping Model
    keypoints = face_mapper.predict(image)[0]

    # Scale Keypoints To Fit Image Dimensions
    keypoints = np.multiply(keypoints, VIDEO_MAPPING_RESOLUTION[0])

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
    # TODO
    pass

def apply_mask():
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
        if (not(fr.shape[0] > 0) or not(fr.shape[1] > 0)):
            # Read Frame From Stream
            ret, fr = stream.read()

            # Skip Frame
            continue

        # Resize Video Frame To Fixed Dimensions
        fr = cv.resize(fr, dsize = VIDEO_CAPTURE_RESOLUTION, interpolation = cv.INTER_AREA)

        # Run Video Frame Through Facial Detector
        faces = detect_face(fr)

        # Iterate Over Faces In Video Frame
        for x, y, w, h in (faces):
            # Copy Frame As Face
            face = fr.copy()

            # Compute Y Coordinates Of Localization Box
            y1 = y - LOCALIZATION_PADDING
            y2 = y + h + LOCALIZATION_PADDING

            # Compute X Coordinates Of Localization Box
            x1 = x - LOCALIZATION_PADDING
            x2 = x + w + LOCALIZATION_PADDING

            # Crop Frame Onto Face
            face = face[y1 : y2, x1 : x2]

            # Verify Face Dimensions
            if (not(face.shape[0] > 0) or not(face.shape[1] > 0)):
                # Skip Face
                continue

            # Run Frame Through Facial Detector
            features = map_face(face)

            # TODO: Scale Points To Original Frame
            ys = y // VIDEO_MAPPING_RESOLUTION[1]
            xs = x // VIDEO_MAPPING_RESOLUTION[0]

            # Map Nose Tip
            face[features["nose_tip"][1] * ys][features["nose_tip"][0] * xs] = 0

            # Map Left Eye
            face[features["left_eye"][1] * ys][features["left_eye"][0] * xs] = 255

            # Map Right Eye
            face[features["right_eye"][1] * ys][features["right_eye"][0] * xs] = 255

            # Map Mouth Center
            face[features["mouth_center"][1] * ys][features["mouth_center"][0] * xs] = 0

            # TODO: Create Masks Using Facial Map Information

            # TODO: Apply Masks To Original Webcam Image

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
