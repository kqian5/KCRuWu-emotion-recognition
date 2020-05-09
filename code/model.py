import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
	Conv2D, MaxPool2D, Dropout, Flatten, Dense, ReLU, Multiply, BatchNormalization


class Model(tf.keras.Model):
	""" Your own neural network model. """

	def __init__(self):
		super(Model, self).__init__()

		# Optimizer
		self.optimizer = tf.keras.optimizers.Adam(
			learning_rate=hp.learning_rate)

		# they used 3x3x10 kernels, so we need to make sure the input is passed in with 10 channels.
		# we can adjust these filter numbers

		self.architecture = [
			# Original vanilla CNN
			# Conv2D(32, 3, 1, padding='same', activation="relu"),
			# Conv2D(32, 3, 1, padding='same'),
			# MaxPool2D(2),
			# ReLU(),

			# Conv2D(64, 3, 1, padding='same', activation="relu"),
			# Conv2D(64, 3, 1, padding='same'),
			# MaxPool2D(2),
			# ReLU(),

			# Dropout(0.4),

			# Flatten(),

			# Dense(40, activation='relu'),
			# Dense(7, activation="softmax")

			# New more complex CNN (Accuracy ~60% after 100 epochs of training)
			Conv2D(64, 3, 1, padding='same', activation="relu"),
			Conv2D(64, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Conv2D(128, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			Conv2D(128, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			Conv2D(128, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Conv2D(256, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			Conv2D(256, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			Conv2D(256, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Conv2D(512, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			Conv2D(512, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			Conv2D(512, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Flatten(),
			Dense(512, activation='relu'),
			Dropout(0.5),
			Dense(256, activation='relu'),
			Dropout(0.5),
			Dense(128, activation='relu'),
			Dropout(0.5),
			Dense(64, activation='relu'),
			Dropout(0.5),

			Dense(7, activation="softmax")
		]


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
