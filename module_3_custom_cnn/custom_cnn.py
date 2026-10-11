import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------
# Parameters
# ---------------------------------------

DATA_DIR = "datasets"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

EPOCHS = 20


# ---------------------------------------
# Load Dataset
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

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names

NUM_CLASSES = len(class_names)


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
# Build Custom CNN
# ---------------------------------------

def build_custom_cnn(num_classes):

    inputs = tf.keras.layers.Input(
        shape=(224, 224, 3)
    )

    x = data_augmentation(inputs)

    x = tf.keras.layers.Rescaling(
        1.0 / 255
    )(x)


    # -----------------------------------
    # Block 1
    # -----------------------------------

    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)


    # -----------------------------------
    # Block 2
    # -----------------------------------

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)


    # -----------------------------------
    # Block 3
    # -----------------------------------

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)


    # -----------------------------------
    # Block 4
    # -----------------------------------

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)


    # -----------------------------------
    # Classification Head
    # -----------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)


    return tf.keras.models.Model(
        inputs,
        outputs
    )


# ---------------------------------------
# Create CNN Model
# ---------------------------------------

cnn_model = build_custom_cnn(NUM_CLASSES)


# ---------------------------------------
# Compile CNN
# ---------------------------------------

cnn_model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ---------------------------------------
# Display Model Architecture
# ---------------------------------------

cnn_model.summary()


# ---------------------------------------
# Train CNN
# ---------------------------------------

history_cnn = cnn_model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS
)


# ---------------------------------------
# Plot Accuracy and Loss
# ---------------------------------------

plt.figure(figsize=(12, 5))


# Accuracy

plt.subplot(1, 2, 1)

plt.plot(
    history_cnn.history["accuracy"]
)

plt.plot(
    history_cnn.history["val_accuracy"]
)

plt.title("CNN Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(
    ["Training", "Validation"]
)


# Loss

plt.subplot(1, 2, 2)

plt.plot(
    history_cnn.history["loss"]
)

plt.plot(
    history_cnn.history["val_loss"]
)

plt.title("CNN Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(
    ["Training", "Validation"]
)

plt.show()