import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------
# Grad-CAM Function
# ---------------------------------------

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_channel = predictions[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()


# ---------------------------------------
# Prepare an Image
# ---------------------------------------

img_path = "sample_leaf.jpg"

img = tf.keras.utils.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = tf.keras.utils.img_to_array(
    img
)

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ---------------------------------------
# Generate Grad-CAM
# ---------------------------------------

heatmap = make_gradcam_heatmap(
    img_array,
    resnet_model,
    "conv5_block3_out"
)


# ---------------------------------------
# Display Grad-CAM
# ---------------------------------------

plt.figure(figsize=(12, 4))


# Original Image

plt.subplot(1, 3, 1)

plt.imshow(img)

plt.title("Original Leaf")

plt.axis("off")


# Grad-CAM Heatmap

plt.subplot(1, 3, 2)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title("Grad-CAM")

plt.axis("off")


# Overlay

plt.subplot(1, 3, 3)

plt.imshow(img)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.4
)

plt.title("Explainable Diagnosis")

plt.axis("off")

plt.show()


# ---------------------------------------
# Final Prediction Function
# ---------------------------------------

def predict_disease(
    image_path,
    model,
    class_names
):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = tf.keras.utils.img_to_array(
        img
    )

    batch = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        batch,
        verbose=0
    )

    class_index = np.argmax(
        predictions[0]
    )

    confidence = predictions[0][class_index]

    disease = class_names[class_index]

    print(
        "Predicted Disease:",
        disease
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    return disease, confidence


# ---------------------------------------
# Call Prediction Function
# ---------------------------------------

predict_disease(
    "sample_leaf.jpg",
    resnet_model,
    class_names
)