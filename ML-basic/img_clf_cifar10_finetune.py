"""
model can be selected based on input for image classification
通过参数选择使用轻量级模型（如MobileNetV2）或ResNet50进行CIFAR-10分类，并支持fine-tuning（微调）
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def build_model(model_name="mobilenet", num_classes=10, fine_tune=False):
    """
    构建一个可选 MobileNetV2 或 ResNet50 预训练模型的分类网络
    参数:
        model_name: 'mobilenet' 或 'resnet'
        num_classes: 分类类别数
        fine_tune: 是否对部分层进行微调
    """
    input_shape = (224, 224, 3)

    if model_name.lower() == "mobilenet":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = mobilenet_preprocess
    elif model_name.lower() == "resnet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = resnet_preprocess
    else:
        raise ValueError("model_name must be 'mobilenet' or 'resnet'")

    # 冻结特征提取层
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess),  # 对输入图像进行预处理
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # 可选 fine-tuning
    if fine_tune:
        base_model.trainable = True
        # 仅解冻靠近顶层的若干层
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    return model


def train_model(model_name="mobilenet", fine_tune=False, epochs=10, batch_size=64):
    # 载入 CIFAR-10 数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 将 CIFAR-10 图片调整为 224x224
    x_train = tf.image.resize(x_train, (224, 224))
    x_test = tf.image.resize(x_test, (224, 224))

    # one-hot 编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # 构建模型
    model = build_model(model_name=model_name, fine_tune=fine_tune)

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=epochs, batch_size=batch_size)

    # 测试评估
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\n✅ Test Accuracy ({model_name}): {test_acc:.4f}")

    return model, history

## test examples:
# 使用轻量级模型（MobileNetV2）
# python train_cifar_pretrained.py --model mobilenet --fine_tune False
#
# 使用 ResNet50 并开启微调
# python train_cifar_pretrained.py --model resnet --fine_tune True

if __name__ == "__main__":
    # 示例：可选 'mobilenet' 或 'resnet'
    model_name = "resnet"  # "mobilenet" 或 "resnet"
    fine_tune = True        # 是否微调顶层

    model, history = train_model(model_name=model_name, fine_tune=fine_tune, epochs=10)
