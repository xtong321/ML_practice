"""
Fine-tune a CNN model based on ResNet
ResNet50 Fine-Tuning（高精度版）
优点：效果更强（≈ 93–95%），适合 GPU
缺点：显存需求较高、训练较慢

只训练分类头（冻结 base）：≈ 85–88%
Fine-tune 后：≈ 92–95%
"""
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ========== 1. 加载数据 ==========
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ResNet50 需要 224×224 输入
x_train = tf.image.resize(x_train, [224, 224])
x_test = tf.image.resize(x_test, [224, 224])

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# ========== 2. 构建模型 ==========
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# ========== 3. 编译 & 初始训练 ==========
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(x_test, y_test)
)

# ========== 4. Fine-Tune（微调 ResNet 后层）==========
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False  # 只解冻最后50层

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_test, y_test)
)


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