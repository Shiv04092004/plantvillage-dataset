import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------
# Parameters
# ---------------------------------------

DATA_DIR = "datasets"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42


# ---------------------------------------
# Data Augmentation
# ---------------------------------------

data_augmentation = tf.keras.Sequential([
    
    tf.keras.layers.RandomFlip("horizontal"),
    
    tf.keras.layers.RandomRotation(0.1),
    
    tf.keras.layers.RandomZoom(0.2),
    
    tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    ),
    
    tf.keras.layers.RandomContrast(0.1)

])


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
# Visualize Augmentation
# ---------------------------------------

for images, labels in train_ds.take(1):

    original = images[0]

    plt.figure(figsize=(10, 8))

    for i in range(9):

        augmented = data_augmentation(
            tf.expand_dims(original, 0),
            training=True
        )

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(
            augmented[0].numpy().astype("uint8")
        )

        plt.axis("off")

    plt.show()


# ---------------------------------------
# Improve Dataset Performance
# ---------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)