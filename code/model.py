"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense, ReLU


class Model(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(Model, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        # they used 3x3x10 kernels, so we need to make sure the input is passed in with 10 channels.
        # we can adjust these filter numbers
        self.architecture = [
            Conv2D(32, 3, 1, padding='same', activation="relu"),
            Conv2D(32, 3, 1, padding='same'),
            MaxPool2D(2),
            ReLU(),

            Conv2D(64, 3, 1, padding='same', activation="relu"),
            Conv2D(64, 3, 1, padding='same'),
            MaxPool2D(2),
            ReLU(),

            Dropout(0.3),
            # insert localization network here
            Flatten(),

            Dense(50, activation='relu'),
            Dense(7, activation="softmax")
        ]

        # ====================================================================

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
