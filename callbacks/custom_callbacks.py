import matplotlib.pyplot as plt
import io
import numpy as np
import sklearn.metrics
import itertools
import tensorflow as tf


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, file_writer):
        super().__init__()
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None):
        class_names = ["one", "two"]

        test_images, test_labels = next(get_validation_generator())

        # Use the model to predict the values from the test_images.
        test_pred_raw = model.predict(test_images)

        test_pred = np.argmax(test_pred_raw, axis=1)
        test_labels = np.argmax(test_labels, axis=1)

        # Calculate the confusion matrix using sklearn.metrics
        cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)

        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


class HelloWorldCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_writer):
        super().__init__()
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None):
        print("Hello world")
