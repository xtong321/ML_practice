"""
Using Keras to Build Custom Object Detection Models
ref: https://www.geeksforgeeks.org/computer-vision/using-keras-to-build-custom-object-detection-models/

Building custom object detection models using Keras (specifically with KerasCV, an extension for Computer Vision tasks) is a powerful way to detect and localize objects in images. Here's a complete guide to understanding, building, training, testing and evaluating a custom object detection model using Keras.

Components of Object Detection Model Working
1. Dataset Loading and Preprocessing
The datasets like Pascal VOC provide images and corresponding bounding box annotations.
Preprocessing includes resizing images, normalizing pixel values and converting bounding box formats.
It ensures consistency and efficient batching for model input.

2. Model Definition
An object detection model predicts both class labels and bounding box coordinates.
Popular architectures include YOLO, RetinaNet and R-CNN.
The model outputs bounding boxes and class probabilities per object.

3. Loss Function and Optimizer
Object detection uses multi-part loss: one for bounding box regression and one for classification.
An optimizer like Adam or SGD minimizes the total loss during training.

4. Training Loop
In each epoch, batches of data are passed through the model. Predictions are compared with ground truths using loss functions and model weights are updated using backpropagation.
Validation data monitors overfitting or underperformance during training.

5. Evaluation Metrics
Model performance is evaluated using metrics like: IoU (Intersection over Union) and mAP (mean Average Precision).
These metrics quantify both localization and classification accuracy.

6. Prediction and Visualization
The trained model is used to predict bounding boxes and classes on new images.
Visualization involves drawing predicted boxes with class labels and confidence scores on the images to visually inspect detection quality.
"""

## 1. Installing necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

## 2. Generating Sample Dataset
# Generate dummy data: 100 images
X = np.random.rand(100, 64, 64, 3)
y = np.random.rand(100, 5)
# Here, we have used randomly generated data of 100 random RGB images of shape 64×64×3. For real use, X would be actual images and y would be bounding box and class labels.

## 3. Model Building
# We use Different layers to form the model. This is a simple Neural Network with two Conv2D layers followed by a Flatten layer and a fully connected Dense output layer. The final Dense(5) layer outputs 5 continuous values without activation. The model uses MSE as the loss parameter.
input_layer = Input(shape=(64, 64, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)

output = Dense(5)(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')  
model.summary()

## 4. Model Training Process
# We set the number of epochs and batch size and Train the built model.
history = model.fit(X, y, epochs=10, batch_size=8)

## 5. Predictions
# This generates a random test image with pixel values between 0 and 1. It runs the trained model on the test image. It extracts the first four values of the prediction and Displays the predicted bounding box coordinates and the rounded class label to simulate classification.
test_image = np.random.rand(1, 64, 64, 3)
prediction = model.predict(test_image)

bbox = prediction[0][:4]
class_label = prediction[0][4]

print("Predicted Bounding Box:", bbox)
print("Predicted Class Label:", round(class_label))

## 6. Evaluation and Visualization
# Here, we are visualizing the computed loss trajectory over the epochs to analyze the Model performance.
plt.figure(figsize=(10, 4))

plt.plot(history.history['loss'], label='Train Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Total Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
# Here, we can analyze that the Loss is decreasing significantly over the Training Process.
