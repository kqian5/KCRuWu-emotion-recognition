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
from SpatialTransformLayer import DefaultLocalizationNetwork
from SpatialTransformLayer import SpatialTransformLayer


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

		# self.vanilla = [
		# 	Conv2D(64, 3, 1, input_shape=(hp.img_size, hp.img_size, 1), padding='same', activation="relu"),
		# 	Conv2D(64, 3, 1, padding='same'),
		# 	MaxPool2D(2),
		# 	ReLU(),

		# 	Conv2D(64, 3, 1, padding='same', activation="relu"),
		# 	Conv2D(64, 3, 1, padding='same'),
		# 	MaxPool2D(2),
		# 	ReLU(),

		# 	Dropout(0.3)
		# ]

		# self.localization = [
		# 	Conv2D(32, 3, 1, input_shape=(hp.img_size, hp.img_size, 1), padding='same'),
		# 	MaxPool2D(2),
		# 	ReLU(),

		# 	Conv2D(32, 3, 1, padding='same'),
		# 	MaxPool2D(2),
		# 	ReLU(),

		# 	Dense(32, activation='relu'),
		# 	Dense(6),
		# ]

		# self.affine = AffineLayer()

		# self.multiply = Multiply()
		self.transformer = SpatialTransformLayer()

		self.head = [

			# insert localization network here
			Flatten(),

			Dense(50, input_shape=(1, )),
			Dense(7, activation="softmax")
		]

	def call(self, img):
		""" Passes input image through the network. """
		# vanilla_out = img
		# for layer in self.vanilla:
		# 	vanilla_out = layer(vanilla_out)
		# out = vanilla_out
		# localization_out = img
		# for layer in self.localization:
		#     localization_out = layer(localization_out)
		# print("localization")
		# print(localization_out.shape)
		# out = transformer(tf.reshape(img, shape=(32, 48, 48, 1)), tf.reshape(localization_out, shape=(-1, 6)))
		# # affine_transformation_layer = self.affine(img, tf.reshape(localization_out, shape = (-1, 2, 3)))
		# print(out.shape)
		# # transformed_vanilla = affine_transformation_layer(vanilla_out)
		out = self.transformer(img)
		for layer in self.head:
		    out = layer(out)
		return out

	@staticmethod
	def loss_fn(labels, predictions):
		""" Loss function for the model. """

		return tf.keras.losses.sparse_categorical_crossentropy(
			labels, predictions, from_logits=False)
