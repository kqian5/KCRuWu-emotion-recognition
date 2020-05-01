"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
	Conv2D, MaxPool2D, Dropout, Flatten, Dense, ReLU, Multiply
from AffineLayer import AffineLayer
from transformer import spatial_transformer_network


class Model(tf.keras.Model):
	""" Your own neural network model. """

	def __init__(self):
		super(Model, self).__init__()

		# Optimizer
		# self.optimizer = tf.keras.optimizers.Adam(
		#     learning_rate=hp.learning_rate,
		#     momentum=hp.momentum)
		self.optimizer = tf.keras.optimizers.Adam(
			learning_rate=hp.learning_rate)

		# they used 3x3x10 kernels, so we need to make sure the input is passed in with 10 channels.
		# we can adjust these filter numbers

		self.architecture = [
			Conv2D(32, 3, 1, input_shape=(hp.img_size, hp.img_size, 1), padding='same', activation="relu"),
			Conv2D(32, 3, 1, padding='same'),
			MaxPool2D(2),
			ReLU(),

			Conv2D(32, 3, 1, padding='same', activation="relu"),
			Conv2D(32, 3, 1, padding='same'),
			MaxPool2D(2),
			ReLU(),

			Dropout(0.3),
			
			Flatten(),

			Dense(50),
			Dense(7, activation="softmax")
		]

		self.localization = [
			Conv2D(32, 3, 1, input_shape=(hp.img_size, hp.img_size, 1), padding='same'),
			MaxPool2D(2),
			ReLU(),

			Conv2D(32, 3, 1, padding='same'),
			MaxPool2D(2),
			ReLU(),
		]

		self.loc_fc = [
			Dense(32, activation='relu'),
			Dense(32),
		]

		# self.multiply = Multiply()

		self.head = [

			# insert localization network here
			
		]

	def call(self, img):
		""" Passes input image through the network. """
		img = tf.convert_to_tensor(img)
		localization_out = img
		for layer in self.localization:
			localization_out = layer(localization_out)

		for layer in self.loc_fc:
			localization_out = layer(localization_out)

		x = spatial_transformer_network(img, localization_out)

		print(x.shape)
		print('finished attention part')
		output = x
		for layer in self.architecture:
			output = layer(output)

		return output

	@staticmethod
	def loss_fn(labels, predictions):
		""" Loss function for the model. """

		return tf.keras.losses.sparse_categorical_crossentropy(
			labels, predictions, from_logits=False)
