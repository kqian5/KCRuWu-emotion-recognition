import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
	Conv2D, MaxPool2D, Dropout, Flatten, Dense, ReLU, Multiply, BatchNormalization
from tensorflow.keras import regularizers
from transformer import spatial_transformer_network


class Model(tf.keras.Model):
	""" Your own neural network model. """

	def __init__(self):
		super(Model, self).__init__()

		# Optimizer
		self.optimizer = tf.keras.optimizers.Adam(
			learning_rate=hp.learning_rate)

		# working complex vanilla CNN architecture
		self.architecture = [
			# New more complex CNN (Accuracy ~60% after 100 epochs of training)
			Conv2D(64, 3, 1, padding='same', activation="relu"),
			Conv2D(64, 3, 1, padding='same', activation="relu"),
			BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Conv2D(128, 3, 1, padding='same', activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			BatchNormalization(),
			Conv2D(128, 3, 1, padding='same', activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			BatchNormalization(),
			# Conv2D(128, 3, 1, padding='same', activation="relu"),
			# BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Conv2D(256, 3, 1, padding='same', activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			BatchNormalization(),
			Conv2D(256, 3, 1, padding='same', activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			BatchNormalization(),
			# Conv2D(256, 3, 1, padding='same', activation="relu"),
			# BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Conv2D(512, 3, 1, padding='same', activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			BatchNormalization(),
			Conv2D(512, 3, 1, padding='same', activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			BatchNormalization(),
			# Conv2D(512, 3, 1, padding='same', activation="relu"),
			# BatchNormalization(),
			MaxPool2D(2),
			Dropout(0.5),

			Flatten(),
			Dense(512, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			Dropout(0.5),
			Dense(256, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			Dropout(0.5),
			Dense(128, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			Dropout(0.5),
			Dense(64, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
			Dropout(0.5),

			Dense(7, activation="softmax", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
		]

		# localizaiton network for attention
		self.localization = [
			Conv2D(32, 3, 1, input_shape=(hp.img_size, hp.img_size, 1), padding='same'),
			MaxPool2D(2),
			ReLU(),

			Conv2D(32, 3, 1, padding='same'),
			MaxPool2D(2),
			ReLU(),
			Dense(32, activation='relu'),
			Dense(32),
		]


	def call(self, img):
		""" Passes input image through the network. """

		'''
		Attempt to applying localization and spatial transformer network
		'''
		# localization_out = img
		# for layer in self.localization:
		# 	localization_out = layer(localization_out)
		# output = spatial_transformer_network(img, localization_out)
		# img = tf.reshape(output, tf.shape(img))

		'''
		Apply CNN architecture
		'''
		for layer in self.architecture:
			img = layer(img)

		return img

	@staticmethod
	def loss_fn(labels, predictions):
		""" Loss function for the model. """

		return tf.keras.losses.sparse_categorical_crossentropy(
			labels, predictions, from_logits=False)
