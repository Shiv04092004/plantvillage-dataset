import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------
# Parameters
# ---------------------------------------

DATA_DIR = "datasets"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

EPOCHS = 20

NUM_CLASSES = None


# ---------------------------------------
# Load the Training Dataset
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
# Load the Validation Dataset
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

print("Classes:")

for i, name in enumerate(class_names):
    print(i, name)

NUM_CLASSES = len(class_names)

print("Number of classes:", NUM_CLASSES)


# ---------------------------------------
# Visualize Original Images
# ---------------------------------------

plt.figure(figsize=(10, 8))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        plt.title(
            class_names[labels[i]]
        )

        plt.axis("off")

plt.show()