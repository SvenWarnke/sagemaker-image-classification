import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from callbacks import custom_callbacks


NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
EPOCHS = 5


def get_data():

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train[:100]
    y_train = y_train[:100]

    x_test = x_test[:10]
    y_test = y_test[:10]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def get_model():

    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def test_doesnt_fail():
    (x_train, y_train), (x_test, y_test) = get_data()
    model = get_model()

    file_writer_cm = tf.summary.create_file_writer('logs/cm')
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            custom_callbacks.HelloWorldCallback(file_writer="bla"),
            custom_callbacks.ConfusionMatrixCallback(file_writer=file_writer_cm, val_data=(x_test, y_test)),
            custom_callbacks.SaveMisslabeledImages(file_writer=file_writer_cm, validation_data=(x_test, y_test))
        ]
    )
