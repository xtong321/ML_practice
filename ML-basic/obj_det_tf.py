"""
Object detection using tensorflow
ref: https://www.geeksforgeeks.org/computer-vision/object-detection-using-tensorflow/

Key Concepts in Object Detection
1) Bounding Boxes: Object detection involves drawing bounding boxes around detected objects. A bounding box is a rectangle that encloses an object and is defined by its coordinates typically, (x_min, y_min) for the top-left corner and (x_max, y_max) for the bottom-right corner.
2) Object Localization: Localization is the process of determining the object's location within the image. It involves predicting the coordinates of the bounding box that encapsulates the object.
3) Class Prediction: Object detection not only locates objects but also categorizes them into different classes (e.g., person, car, dog). Each object is assigned a class label, providing information about what the object is.
4) Model Architectures: Numerous architectures are used for object detection, such as SSD (Single Shot Multibox Detector), Faster R-CNN (Region-based Convolutional Neural Network) and YOLO (us Only Look Once). These models differ in their approach to balancing speed and accuracy.
"""

## Step-by-Step Object Detection using TensorFlow

## Step 1: Install and Import Libraries
# Let's import the necessary libraries,
#- tensorflow as tf: Core library used for machine learning; handles loading and running pre-trained models.
#- numpy as np: Fundamental array-manipulation package; converts images to arrays and processes detection outputs.
#- cv2: OpenCV, for image processing tasks like drawing rectangles and text on images.
#- from PIL import Image: From Pillow; used for opening images and converting them to RGB.
#- from matplotlib import pyplot as plt: For displaying the final images with bounding boxes in Colab notebooks.
#- from random import randint: Generates random colors for bounding boxes so each detected object’s box is visually distinct.

#pip install -U tensorflow
#pip install opencv-python-headless pillow matplotlib
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from random import randint

## Step 2: Download, Extract and Load the Pre-trained Model
# Now, load the pre-trained model using TensorFlow's SavedModel format.
# !wget : Downloads the pre-trained object detection model file (SSD MobileNet v2 variant) from TensorFlow’s servers.
# !tar -xzf : Extracts the downloaded .tar.gz archive so we can access the model files.

# !wget http: // download.tensorflow.org / models / object_detection / tf2 / 20200711 / ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu - 8.tar.gz
# !tar - xzf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu - 8.tar.gz
model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model")

## Step 3: Load and Preprocess Image
# Upload an image, convert it to a NumPy array and preprocess it for input to the model, as the model can't directly work on an image therefore we first converted it into a tensor.
# from google.colab import files: Imports Colab utility functions for uploading files interactively.
# files.upload(): Lets us upload a local image directly into the Colab environment.
# Image.open(image_path): Opens the image using Pillow.
# .convert('RGB'): Ensures the image is in RGB color mode (3 channels), which is required by the detection model.
# np.expand_dims(..., 0): Adds a new dimension to the array for “batch size” (shape becomes [1, height, width, 3]).
# tf.convert_to_tensor(..., dtype=tf.uint8): Converts the array to a TensorFlow tensor, using the 8-bit integer format expected by the model.
from google.colab import files
uploaded = files.upload()

image_path = list(uploaded.keys())[0]
image = Image.open(image_path).convert('RGB')
image_np = np.array(image)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

## Step 4: Perform Object Detection
#Now we perform the object detection,
#
# detection['detection_boxes']: Contains the bounding box coordinates (normalized between 0 and 1).
# .numpy(): Converts the tensor to a NumPy array for ease of manipulation.
# detection['detection_classes']: Gives the class indices for each detected object; .astype(int) ensures these can be used as indices in the label list.
# detection['detection_scores']: Contains the confidence scores for each detection.
detection = model(input_tensor)
boxes = detection['detection_boxes'].numpy()
classes = detection['detection_classes'].numpy().astype(int)
scores = detection['detection_scores'].numpy()

## Step 5: Run Object Detection
# These are the labels for the COCO dataset, which contains class names corresponding to class IDs.
#
# Defines the list of all class names in the COCO dataset.
# Allows translating class indices from model output into human-readable names.
labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

## Step 6: Visualize the detected objects
confidence_threshold = 0.5
h, w, _ = image_np.shape

for i in range(classes.shape[1]):
    class_id = int(classes[0, i])
    score = scores[0, i]
    if score > confidence_threshold:
        ymin, xmin, ymax, xmax = boxes[0, i]
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)
        class_name = labels[class_id]

        color = (randint(0, 256), randint(0, 256), randint(0, 256))
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image_np, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

plt.imshow(image_np)
plt.axis('off')
plt.show()