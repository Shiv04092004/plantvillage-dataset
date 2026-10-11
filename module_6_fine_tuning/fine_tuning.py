import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)


# ---------------------------------------
# Parameters
# ---------------------------------------

DATA_DIR = "datasets"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42


# ---------------------------------------
# Load Training Dataset
# ---------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)


# ---------------------------------------
# Load Validation Dataset
# ---------------------------------------

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)


# ---------------------------------------
# Get Class Names
# ---------------------------------------

class_names = train_ds.class_names

NUM_CLASSES = len(class_names)

print("Classes:")

for i, name in enumerate(class_names):
    print(i, name)

print("Number of classes:", NUM_CLASSES)


# ---------------------------------------
# Data Augmentation
# ---------------------------------------

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.2),

    layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    ),

    layers.RandomContrast(0.1)

])


# ---------------------------------------
# Load Pretrained ResNet50
# ---------------------------------------

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ---------------------------------------
# Initially Freeze ResNet50
# ---------------------------------------

base_model.trainable = False


# ---------------------------------------
# Build Model
# ---------------------------------------

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = resnet_preprocess(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

resnet_model = models.Model(
    inputs,
    outputs
)


# ---------------------------------------
# Initial Training
# ---------------------------------------

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)


# ---------------------------------------
# Fine-Tuning
# ---------------------------------------

base_model.trainable = True


# Freeze earlier layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


# ---------------------------------------
# Recompile with Small Learning Rate
# ---------------------------------------

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------------------
# Fine-Tune
# ---------------------------------------

history_resnet_ft = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)