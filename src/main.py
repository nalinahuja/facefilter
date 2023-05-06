# Written By Nalin Ahuja (nalinahuja), Rohan Narasimha (novablazerx)

import os
import sys

# Set TensorFlow Logging Level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# End Imports-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Print Format Strings
NL, TB, CR, CL = "\n", "\t", "\r", "\033[2K"

# Embedded Resource Paths
IMAGE_MASK_PATH = "./masks"
FACE_MAPPING_MODEL = "./mapping"
FACE_DETECTION_MODEL = "./detection"

# Video Capture Resolution (px, px)
VIDEO_CAPTURE_RESOLUTION = (1280, 720)

# Model Mapping Resolution (px, px)
MODEL_MAPPING_RESOLUTION = (96, 96)

# Selected Mask Filename
SELECTED_MASK = "glasses.png"

# End Embedded Constants------------------------------------------------------------------------------------------------------------------------------------------------

import random
import cv2 as cv
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn import metrics
from tensorflow import keras

# TensorFlow Settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# End Module Imports----------------------------------------------------------------------------------------------------------------------------------------------------

# Print Status
print(NL + CR + "Loading facial detection model...", end = "")

try:
    # Load Face Detection Model
    face_detector = cv.CascadeClassifier(os.path.join(FACE_DETECTION_MODEL, "saved", "detection_model.xml"))
except Exception as e:
    # Raise FileNotFoundError
    raise FileNotFoundError("could not load facial detection model")

def detect_faces(image):
    # Convert Image Colorspace To Grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Localize Faces In Image
    faces = face_detector.detectMultiScale(image, scaleFactor = 1.30, minNeighbors = 5, minSize = MODEL_MAPPING_RESOLUTION, flags = cv.CASCADE_SCALE_IMAGE)

    # Return Facial Coordinates
    return (faces)

# End Facial Detection Model--------------------------------------------------------------------------------------------------------------------------------------------

# Print Status
print(CL + CR + "Loading facial mapping model...", end = "")

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
    image = cv.resize(image, dsize = MODEL_MAPPING_RESOLUTION)

    # Normalize Image Pixel Values
    image = np.divide(image, 255)

    # Expand Pixel Value Dimension
    image = np.expand_dims(image, axis = -1)

    # Expand Image Frame Dimension
    image = np.expand_dims(image, axis = 0)

    # Run Image Through Facial Mapping Model
    keypoints, *_ = face_mapper.predict(image)

    # Scale Keypoints To Fit Image Dimensions
    keypoints = np.multiply(keypoints, w)

    # Round Keypoint Coordinates
    keypoints = np.round(keypoints)

    # Cast Keypoint Values
    keypoints = np.int32(keypoints)

    # Initialize Feature Dictionary
    features = {}

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

# End Facial Mapping Model----------------------------------------------------------------------------------------------------------------------------------------------

# Print Status
print(CL + CR + f"Loading '{SELECTED_MASK}' image mask...", end = "")

# Form Mask Path
mask_path = os.path.join(IMAGE_MASK_PATH, SELECTED_MASK)

# Verify Mask Path
if (not(os.path.exists(mask_path))):
    # Raise FileNotFoundError
    raise FileNotFoundError("image mask does not exist")

# Initialize Global Image Mask
mask = cv.imread(mask_path, -1)

# Conditionally Initialize Overlay Function
if (SELECTED_MASK == "glasses.png"):
    # Initialize Mask Size
    mask_size = (400, 80)

    # Initialize Mask Anchors
    mask_anchors = {"left_eye": (100, 40), "right_eye": (300, 40)}

    # Calculate Mask Anchor X Delta
    mask_x_delta = abs(mask_anchors["left_eye"][0] - mask_anchors["right_eye"][0])

    def overlay_mask(fr, features):
        # Set Mask Scope To Global
        global mask

        # Get Left Eye Feature Coordinates
        left_eye_x, left_eye_y = features["left_eye"]

        # Get Right Eye Feature Coordinates
        right_eye_x, right_eye_y = features["right_eye"]

        # Calculate Difference Between X Components Of Eye Coordinates
        eye_x_delta = abs(left_eye_x - right_eye_x)

        # Calculate Image Width Scaling Factor
        w_scaler = float(eye_x_delta / mask_x_delta) * 1.25

        # Calculate Mask Width
        mask_w = int((mask_size[0]) * w_scaler)

        # Calculate Mask Height
        mask_h = int((mask_size[1] / mask_size[0]) * mask_w)

        # Convert Video Frame To PIL
        fr = Image.fromarray(fr)

        # Convert Fit Mask To PIL
        fit_mask = Image.fromarray(cv.resize(mask, dsize = (mask_w, mask_h)))

        # Calculate Mask X Coordinate
        mx = int(left_eye_x - eye_x_delta - (w_scaler * (mask_anchors["left_eye"][0] + 20)))

        # Calculate Mask Y Coordinate
        my = left_eye_y - 10

        # Overlay Mask On Image Frame
        fr.paste(fit_mask, (mx, my), fit_mask)

        # Convert Video Frame To OpenCV
        fr = np.array(fr)

        # Return Modified Image Frame
        return (fr)

elif (SELECTED_MASK == "santa.png"):
    # Initialize Mask Size
    mask_size = (820, 570)

    # Initialize Mask Anchors
    mask_anchors = {"left_eye": (435, 480), "right_eye": (700, 420)}

    # Calculate Mask Anchor X Delta
    mask_x_delta = abs(mask_anchors["left_eye"][0] - mask_anchors["right_eye"][0])

    def overlay_mask(fr, features):
        # Set Mask Scope To Global
        global mask

        # Get Left Eye Feature Coordinates
        left_eye_x, left_eye_y = features["left_eye"]

        # Get Right Eye Feature Coordinates
        right_eye_x, right_eye_y = features["right_eye"]

        # Calculate Difference Between X Components Of Eye Coordinates
        eye_x_delta = abs(left_eye_x - right_eye_x)

        # Calculate Image Width Scaling Factor
        w_scaler = float(eye_x_delta / mask_x_delta) * 1.25

        # Calculate Mask Width
        mask_w = int((mask_size[0]) * w_scaler)

        # Calculate Mask Height
        mask_h = int((mask_size[1] / mask_size[0]) * mask_w)

        # Convert Video Frame To PIL
        fr = Image.fromarray(fr)

        # Convert Fit Mask To PIL
        fit_mask = Image.fromarray(cv.resize(mask, dsize = (mask_w, mask_h)))

        # Calculate Mask X Coordinate
        mx = int(left_eye_x - eye_x_delta - (w_scaler * (mask_anchors["left_eye"][0] + 20)))

        # Calculate Mask Y Coordinate
        my = left_eye_y - mask_h - 25

        # Overlay Mask On Image Frame
        fr.paste(fit_mask, (mx, my), fit_mask)

        # Convert Video Frame To OpenCV
        fr = np.array(fr)

        # Return Modified Image Frame
        return (fr)

elif (SELECTED_MASK == "saiyan.png"):
    # Initialize Mask Size
    mask_size = (648, 760)

    # Initialize Mask Anchors
    mask_anchors = {"left_eye": (236, 611), "right_eye": (404, 611)}

    # Calculate Mask Anchor X Delta
    mask_x_delta = abs(mask_anchors["left_eye"][0] - mask_anchors["right_eye"][0])

    def overlay_mask(fr, features):
        # Set Mask Scope To Global
        global mask

        # Get Left Eye Feature Coordinates
        left_eye_x, left_eye_y = features["left_eye"]

        # Get Right Eye Feature Coordinates
        right_eye_x, right_eye_y = features["right_eye"]

        # Calculate Difference Between X Components Of Eye Coordinates
        eye_x_delta = abs(left_eye_x - right_eye_x)

        # Calculate Image Width Scaling Factor
        w_scaler = float(eye_x_delta / mask_x_delta)

        # Calculate Mask Width
        mask_w = int((mask_size[0]) * w_scaler * 0.80)

        # Calculate Mask Height
        mask_h = int((mask_size[1] * 0.80 / mask_size[0]) * mask_w)

        # Convert Video Frame To PIL
        fr = Image.fromarray(fr)

        # Convert Fit Mask To PIL
        fit_mask = Image.fromarray(cv.resize(mask, dsize = (mask_w, mask_h)))

        # Calculate Mask X Coordinate
        mx = int(left_eye_x - eye_x_delta - (w_scaler * (mask_anchors["left_eye"][0] - 80)))

        # Calculate Mask Y Coordinate
        my = left_eye_y - mask_h

        # Overlay Mask On Image Frame
        fr.paste(fit_mask, (mx, my), fit_mask)

        # Convert Video Frame To OpenCV
        fr = np.array(fr)

        # Return Modified Image Frame
        return (fr)

elif (SELECTED_MASK == "flame.png"):
    # Initialize Mask Size
    mask_size = (750, 1023)

    # Initialize Mask Anchors
    mask_anchors = {"nose_tip": (360, 480), "mouth_center": (420, 934)}

    # Calculate Mask Anchor X Delta
    mask_y_delta = abs(mask_anchors["nose_tip"][1] - mask_anchors["mouth_center"][1])

    def overlay_mask(fr, features):
        # Set Mask Scope To Global
        global mask

        # Get Nose Tip Feature Coordinates
        nose_tip_x, nose_tip_y = features["nose_tip"]

        # Get Mouth Center Feature Coordinates
        mouth_center_x, mouth_center_y = features["mouth_center"]

        # Calculate Difference Between Y Components Of Face Coordinates
        face_y_delta = abs(nose_tip_y - mouth_center_y)

        # Calculate Image Width Scaling Factor
        w_scaler = float(face_y_delta / mask_y_delta)

        # Calculate Mask Width
        mask_w = int((mask_size[0]) * w_scaler)

        # Calculate Mask Height
        mask_h = int((mask_size[1] / mask_size[0]) * mask_w)

        # Convert Video Frame To PIL
        fr = Image.fromarray(fr)

        # Convert Fit Mask To PIL
        fit_mask = Image.fromarray(cv.resize(mask, dsize = (mask_w, mask_h)))

        # Calculate Mask X Coordinate
        mx = int(mouth_center_x - (w_scaler * (mask_anchors["mouth_center"][0])))

        # Calculate Mask Y Coordinate
        my = mouth_center_y - mask_h + 65

        # Overlay Mask On Image Frame
        fr.paste(fit_mask, (mx, my), fit_mask)

        # Convert Video Frame To OpenCV
        fr = np.array(fr)

        # Return Modified Image Frame
        return (fr)

# End Image Masking Functions-------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == "__main__"):
    # Print Status
    print(CL + CR + f"Starting video capture with mask '{SELECTED_MASK}'..." + NL)

    # Create Video Stream
    stream = cv.VideoCapture(0)

    # Read Frame From Stream
    ret, fr = stream.read()

    try:
        # Process Frames
        while (ret):
            # Verify Frame Dimensions
            if ((fr.shape[0] > 0) and (fr.shape[1] > 0)):
                # Resize Video Frame To Fixed Dimensions
                fr = cv.resize(fr, dsize = VIDEO_CAPTURE_RESOLUTION)

                # Run Video Frame Through Facial Detector
                faces = detect_faces(fr)

                # Iterate Over Faces In Video Frame
                for x, y, w, h in (faces):
                    # Extract Region Of Interest
                    face = fr[y : y + h, x : x + w]

                    # Verify Face Dimensions
                    if ((face.shape[0] > 0) and (face.shape[1] > 0)):
                        # Run Region Of Interest Through Facial Mapper
                        features = map_face(face, x, y, w, h)

                        # Overlay Image Masks Using Feature Coordinates
                        fr = overlay_mask(fr, features)

                # Show Video Frame
                cv.imshow("FaceFilter", cv.flip(fr, 1))

                # Check For Escape Key
                if (cv.waitKey(1) == 27):
                    # Break Loop
                    break

            # Read Frame From Stream
            ret, fr = stream.read()
    except KeyboardInterrupt:
        # Release Frame Stream
        stream.release()

# End Main--------------------------------------------------------------------------------------------------------------------------------------------------------------
