"""
Implement an image classification algo with CNN
ref: https://www.geeksforgeeks.org/machine-learning/image-classifier-using-cnn/

Key Components of CNNs
A Convolutional Neural Network (CNN) is made up of several layers, each designed to perform a specific function in processing images:

1) Convolutional Layers: Filters or kernels that detect features such as edges or textures.
2) ReLU Activation: Adds non-linearity, helping the model learn complex patterns.
3) Pooling Layers: Reduce the dimensions of the image making the network more efficient while preserving important features.
4) Fully Connected Layers: After feature extraction, these layers make the final prediction based on the detected patterns.
5) Softmax Output: Converts the network’s output into probabilities, showing the likelihood of each class.

How CNNs Work for Image Classification?
The process of image classification with a CNN involves several stages:

1) Preprocessing the Image: Images need to be preprocessed before feeding them into the CNN. This includes resizing, normalizing and sometimes augmenting images to make the model more robust and reduce overfitting.
2) Feature Extraction: CNNs automatically detect features from images, starting with simple features like edges and progressing to more complex patterns like objects or faces as we go deeper into the network.
3) Classification: After extracting features, the fully connected layers use the learned information to classify the image. Based on the features detected, the model assigns the image to one of the predefined categories.
"""

txf_data_arg = True
txf_new_cnn = True
txf_new_opt= True

## step-1: Importing Libraries
# We will be using Tensorflow and Matplotlib libraries for building, training and visualizing training and validation accuracy of the model.
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

## Step 2: Downloading and Preparing the Dataset
# Next we load the CIFAR-10 dataset and preprocess it. It consists of 60,000 32x32 color images across 10 categories.
# Scaling: We scale the image pixel values from [0, 255] to [0, 1] by dividing by 255.
# One-hot encoding: Converts the labels (0-9) into a one-hot vector (e.g., for label 2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test , num_classes)

#data argumentation
if txf_data_arg:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)


## Step 3: Building the CNN Model
# Now, we define the CNN architecture and start with convolutional layers followed by max-pooling layers, flatten the output and then feed it into fully connected layers.
# Flatten Layer: Converts the 2D matrix into a 1D vector for the dense layers.
# Dense Layers: Fully connected layers used for decision making, with the final layer using softmax activation to predict probabilities.
"""
model = models.Sequential([    
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    #layers.BatchNormalization(),    
    layers.MaxPooling2D(2,2),
    #layers.Dropout(0.5),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    #layers.BatchNormalization(),    
    layers.MaxPooling2D(2,2),
    #layers.Dropout(0.5),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    #layers.BatchNormalization(),
    #layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
"""
# improved network architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()


## Step 4: Compiling and Training the Model
# We then compile the model by defining the optimizer, loss function and evaluation metric, followed by training. Adam optimizer is used as it adjusts the learning rate during training.
if txf_new_opt:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
else:
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=15,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=2)

## Step 5: Evaluating the Model
# After training, we evaluate the model on the test dataset to check how well it performs on unseen data.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy = {test_acc:.3f}")

## Step 6: Plotting of Accuracy Curves
# Finally, we visualize the training and validation accuracy during training using matplotlib.
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title('Accuracy')
plt.show()