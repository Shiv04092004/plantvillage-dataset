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

EPOCHS = 20


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
# Build ResNet50
# ---------------------------------------

def build_resnet50(num_classes):

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained ResNet50 layers
    base_model.trainable = False

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    # Data augmentation
    x = data_augmentation(inputs)

    # ResNet50 preprocessing
    x = resnet_preprocess(x)

    # Feature extraction
    x = base_model(
        x,
        training=False
    )

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return models.Model(
        inputs,
        outputs
    ), base_model


# ---------------------------------------
# Create ResNet50 Model
# ---------------------------------------

resnet_model, resnet_base = build_resnet50(
    NUM_CLASSES
)


# ---------------------------------------
# Compile ResNet50
# ---------------------------------------

resnet_model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ---------------------------------------
# Train ResNet50
# ---------------------------------------

history_resnet = resnet_model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS
)