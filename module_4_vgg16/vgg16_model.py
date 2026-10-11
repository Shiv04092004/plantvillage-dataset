import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg_preprocess
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
# Build VGG16
# ---------------------------------------

def build_vgg16(num_classes):

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze VGG16 layers
    base_model.trainable = False

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    # Data augmentation
    x = data_augmentation(inputs)

    # VGG16 preprocessing
    x = vgg_preprocess(x)

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
# Create VGG16 Model
# ---------------------------------------

vgg_model, vgg_base = build_vgg16(
    NUM_CLASSES
)


# ---------------------------------------
# Compile VGG16
# ---------------------------------------

vgg_model.compile(
    
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    
    loss="sparse_categorical_crossentropy",
    
    metrics=["accuracy"]
)


# ---------------------------------------
# Train VGG16
# ---------------------------------------

history_vgg = vgg_model.fit(
    
    train_ds,
    
    validation_data=val_ds,
    
    epochs=EPOCHS
)