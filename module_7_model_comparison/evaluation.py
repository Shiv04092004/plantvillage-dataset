import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ---------------------------------------
# Evaluate the Models
# ---------------------------------------

def evaluate_models(
    cnn_model,
    vgg_model,
    resnet_model,
    val_ds,
    class_names
):

    # -----------------------------------
    # Accuracy
    # -----------------------------------

    cnn_loss, cnn_acc = cnn_model.evaluate(val_ds)

    vgg_loss, vgg_acc = vgg_model.evaluate(val_ds)

    resnet_loss, resnet_acc = resnet_model.evaluate(val_ds)

    print("CNN Accuracy:", cnn_acc)

    print("VGG16 Accuracy:", vgg_acc)

    print("ResNet50 Accuracy:", resnet_acc)


    # -----------------------------------
    # Generate Predictions
    # -----------------------------------

    def get_predictions(model, dataset):

        y_true = []

        y_pred = []

        for images, labels in dataset:

            predictions = model.predict(
                images,
                verbose=0
            )

            predictions = np.argmax(
                predictions,
                axis=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions
            )

        return (
            np.array(y_true),
            np.array(y_pred)
        )


    # -----------------------------------
    # Predictions from ResNet50
    # -----------------------------------

    y_true, y_pred = get_predictions(
        resnet_model,
        val_ds
    )


    # -----------------------------------
    # Precision
    # -----------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )


    # -----------------------------------
    # Recall
    # -----------------------------------

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )


    # -----------------------------------
    # F1 Score
    # -----------------------------------

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("Precision:", precision)

    print("Recall:", recall)

    print("F1 Score:", f1)


    # -----------------------------------
    # Classification Report
    # -----------------------------------

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )


    # -----------------------------------
    # Confusion Matrix
    # -----------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()