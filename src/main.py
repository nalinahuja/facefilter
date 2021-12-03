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

# Video Mapping Resolution (px, px)
MODEL_MAPPING_RESOLUTION = (96, 96)

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

    # Extract Image Dimension Before Resizing
    dim = image.shape[0]

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
    keypoints = np.multiply(keypoints, dim)

    # Round Keypoint Coordinates
    keypoints = np.round(keypoints)

    # Cast Keypoint Values
    keypoints = keypoints.astype(np.int32)

    # Initialize Feature Dictionary
    features = dict()

    # Set Nose Tip Coordinates
    features["nose_tip"] = keypoints[0 : 2]

    # Set Left Eye Coordinates
    features["left_eye"] = keypoints[2 : 4]

    # Set Right Eye Coordinates
    features["right_eye"] = keypoints[4 : 6]

    # Set Mouth Center Coodinates
    features["mouth_center"] = keypoints[6 : 8]

    # Return Featues Dictionary
    return (features)

# End Facial Mapping Model Loading---------------------------------------------------------------------------------------------------------------------------------------

overlay = cv.imread("./masks/glasses.png", -1)

def overlay_transparent(background, x, y):
    global overlay
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

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
        if (fr.shape[0] > 0 and fr.shape[1] > 0):
            # Resize Video Frame To Fixed Dimensions
            fr = cv.resize(fr, dsize = VIDEO_CAPTURE_RESOLUTION, interpolation = cv.INTER_AREA)

            # Run Video Frame Through Facial Detector
            faces = detect_face(fr)

            # Define Features Dictionary
            features = None

            # Iterate Over Faces In Video Frame
            for x, y, w, h in (faces):
                # Copy Frame As Face
                face = fr.copy()

                # Crop Frame Into Face
                face = face[y : y + h, x : x + w]

                # Verify Face Dimensions
                if (not(face.shape[0] > 0) or not(face.shape[1] > 0)):
                    # Skip Face
                    continue

                # Run Frame Through Facial Detector
                features = map_face(face)

                # Iterate Over Feature Keys
                for key in (features):
                    # Update Feature X Coordinate
                    features[key][0] += x

                    # Update Feature Y Coordinate
                    features[key][1] += y

                # Apply Features To Masks
                fr = overlay_transparent(fr, features["nose_tip"][0] - 200, features["nose_tip"][1] - 100)

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
