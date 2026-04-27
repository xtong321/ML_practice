"""
detect object using haar-cascade algorithm
ref: https://www.geeksforgeeks.org/python/detect-an-object-with-opencv-python/

Object detection refers to identifying and locating objects within images or videos. OpenCV provides a simple way to implement object detection using Haar Cascades a classifier trained to detect objects based on positive and negative images. In this article we will focus on detecting objects using it which is simple and effective for real-time object detection. But before that lets understand what is haar cascades.

Understanding Haar Cascades
Haar Cascade classifiers are an effective tool for object detection based on Haar features. The key idea is to detect objects in images at multiple scales using a cascade of classifiers. It was originally introduced by Paul Viola and Michael Jones in their paper.

Positive Images: These are images that contain the object to be detected like a stop sign.
Negative Images: These are images that do not contain the object and are used to train the classifier to learn what to ignore.
Haar Cascades are particularly well-suited for real-time object detection making them a popular choice for detecting faces, vehicles and stop signs.

Implementation of Object Detection with Haar Cascades
Here is the step by step implementation of object detection using OpenCV. For this you can download the Haar Cascade XML file for object detection and the sample image from here. Place them in the same directory as your Python script.
"""

## 1. Loading the Image
# The first step in object detection is to load the image in which you want to detect objects.
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")

## 2. Converting Image Color Formats
# OpenCV reads images in BGR format by default but we need the image in RGB format for better visualization with Matplotlib. Additionally for object detection we need a grayscale version of the image.
# cv2.cvtColor(): Converts the image from one color space to another (BGR to RGB and BGR to grayscale).
# cv2.COLOR_BGR2GRAY: Converts an image from BGR (Blue, Green, Red) color space to grayscale.
# cv2.COLOR_BGR2RGB: Converts an image from BGR (Blue, Green, Red) color space to RGB (Red, Green, Blue) color space.
# plt.imshow(): Displays the image in a Matplotlib window.
# plt.show(): Renders the image.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_rgb)
plt.show()

## 3. Loading Haar Cascade Classifier
# Next you need to load the Haar Cascade classifier. The classifier is trained to detect specific objects such as stop signs in this case. Ensure that the XML file is present in your project directory.
# cv2.CascadeClassifier(): Loads the Haar Cascade XML file that contains the trained classifier.
stop_cascade = cv2.CascadeClassifier('stop_data.xml')

## 4. Detecting Objects in the Image
# 1) detectMultiScale(): detect objects in the image. This function performs detection at multiple scales which helps detect objects of varying sizes in grayscale.
# 2) minSize=(20, 20): Defines the minimum size of the object to be detected. Smaller objects are ignored.
found = stop_cascade.detectMultiScale(img_gray, minSize=(20, 20))

## 5. Drawing Rectangles Around Detected Objects
# Once objects are detected we draw a green rectangle around each detected object. This helps visualize the detection result.
# cv2.rectangle(): Draws a rectangle on the image at the coordinates (x, y) with a width w and height h. The color is green (0, 255, 0) and the rectangle has a thickness of 5 pixels.
for (x, y, w, h) in found:
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 5)

## 6. Displaying the Result
# Finally, the processed image is displayed using Matplotlib, showing the detected objects with rectangles.
plt.imshow(img_rgb)
plt.show()